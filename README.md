# DiscoveryHub pipeline_2 — Retrieval-Quality R&D (P0 + P1 critical path)

An **additive** pipeline that implements the highest-leverage part of the OpenAI tech
spec: the **P0 (pooled multi-positive evaluation)** and **P1 (multi-teacher, multi-positive
distillation bakeoff)** critical path. It reuses the existing `discovery_hub` package
(one source of truth for corpus/serving config) and runs the compute-heavy work on a
rented Vast H100, then pulls results back to Drew — **unattended**.

This is **post-demo R&D**: a full run (candidate pooling → cross-encoder scoring → LLM
adjudication → objective bakeoff → train winner → re-embed/eval) is many GPU-hours and
cannot finish before the July 17 demo. Build now, run after.

> **Teacher design (revised 2026-07-15):** the two cross-encoders are **not
> interchangeable and must not be averaged**. Over 55,605 adjudicated pairs,
> BGE-reranker-v2-m3 correlates **+0.468 with word overlap** and only **+0.211 with the
> LLM's utility grade**; Qwen3-Reranker-8B is **+0.371 / +0.410**. At fixed relevance
> (grade 3) BGE's score swings **4.1×** on wording alone, Qwen's **1.15×**. BGE is a
> lexical matcher, so it is kept as an explicitly-labelled lexical channel and **Qwen is
> the primary semantic judge**. **glm-4.6 (via z.ai)** is the gray-zone judge for teacher
> disagreements. Because human calibration is replaced by the LLM, all outputs are framed
> as **"LLM-adjudicated, not human-calibrated"** (spec S2.4 / risk register).
>
> The 60% "gray zone" was never disagreement: `|bge − qwen| > 0.25` fires on **every true
> positive**, because BGE says ~0.05 where Qwen says ~0.80. One teacher measures strings,
> the other measures meaning, on a corpus where those diverge 76% of the time.

---

## What it produces

- `qrels_exact_origin_v1.jsonl` — continuity benchmark (designated positives)
- `qrels_scout_utility_v1.jsonl` — graded 0–3 multi-positive utility qrels
- `multi_positive_labels_v1.jsonl` — merged teacher+LLM graded training labels
- per-arm eval reports (`eval_*.md/json`) with exact-origin + graded-utility metrics and
  promotion-gate checks
- the **winning 4B model** (merged, validated) if a bakeoff arm beats the 4B baseline

---

## Layout

```
discovery_hub_pipeline_2/
├── dh2/                       # the new package (imports discovery_hub/*)
│   ├── config2.py             # paths/knobs; reuses base config; promotion gates (S5.2)
│   ├── llm_client.py          # glm-4.6 via z.ai (retry/cache/mask)
│   ├── retriever2.py          # load ANY (model, vectors, ids) for candidate pooling
│   ├── candidate_pool.py      # union: untouched-0.6B + 8B + 4B + exact-ID; round-robin cap
│   ├── teachers.py            # BgeTeacher (512) + QwenTeacher (official prompt, 8K, margin)
│   ├── teacher_merge.py       # agreement → gray-zone LLM → graded label / mask
│   ├── eval_pool_build.py     # exact + graded qrels; held-out split; family-collapse
│   ├── graded_metrics.py      # nDCG/Success/Recall@100, LOW-HIGH buckets, oracle, CIs
│   ├── losses.py              # MarginMSE, listwise-KL, LSEPair (+ torch modules)
│   ├── train_bakeoff.py       # arms A–F; screen on 0.6B; train winner on 4B (safe merge)
│   ├── validate.py            # LoRA-merge FIX + startup dim/prefix/index checks (S10)
│   ├── manifest.py            # §9.2 version metadata + doc-order checksum
│   ├── cascade.py             # ★ RASC: union → Qwen rerank → optional GroupRank
│   ├── grouprank.py           # ★ Diver-GroupRank-32B via vLLM (TP=4), fail-closed
│   ├── doc_views.py           # ★ source-aware compact document views
│   ├── identifiers.py         # ★ routed NCT/patent/CAS/compound channel (no BM25)
│   └── parallel_tracks_stubs.py  # P2–P8 interface stubs (documented, not implemented)
├── stages/                    # thin runnables (argparse → dh2/*)
│   ├── p0_0_embed_channel.py  # ★ embed corpus with a channel model (RUN THIS FIRST)
│   ├── p0_1_build_candidate_pool.py
│   ├── p0_2_score_teachers.py
│   ├── p0_3_llm_adjudicate.py
│   ├── p0_4_build_eval_qrels.py
│   ├── p1_1_build_train_labels.py
│   ├── p1_2_run_bakeoff.py
│   ├── p1_3_eval_and_compare.py
│   └── p1_4_eval_cascade.py   # ★ C0/C1/C2/C3 + oracle + promotion gates
├── prod/
│   └── 07_retrieve_rank.py    # ★ production serving, cascade behind a feature flag
├── scripts/
│   ├── push_pipeline2_to_h100.sh   # Drew → H100 (code+data+models+env)
│   ├── run_pipeline2_h100.sh       # THE unattended orchestrator (writes DONE sentinel)
│   ├── pull_pipeline2_results.sh   # H100 → Drew (explicit tarball; fails loud)
│   └── watchdog.sh                 # Drew-side: poll DONE → pull → destroy box
├── tests/
│   ├── test_dh2_smoke.py           # mini-corpus end-to-end (no GPU/net/torch)
│   ├── test_losses_bugfix.py       # numerical checks for loss bugs #3/#4/#5 (torch CPU)
│   └── test_07_cascade_flags.py    # flag-off == the untouched demo path
├── analysis/register_gap_analysis.py
├── requirements2.txt
└── README.md
```

★ = added/rewritten 2026-07-15 for the Register-Aware Semantic Cascade.

---

## The Register-Aware Semantic Cascade (RASC)

**Do not retrain.** The bakeoff's finding is that fine-tuning on these pairs teaches
lexical matching: the best arm gained +0.165 R@10 on high-overlap queries and lost 0.089
on low-overlap ones, and 76% of this corpus's judged true matches are low-overlap. The
synthetic queries are generated *from* their origin document, so the training pairs don't
contain the register gap and neither does the benchmark. Another 4B/8B run optimizes the
same artifact.

What ships instead is inference-only:

```
scout query
    ├── untouched Qwen3-Embedding-0.6B  top 100    semantic hedge (best low-overlap arm)
    ├── fine-tuned 8B                   top 100    domain/lexical specialist
    ├── fine-tuned 4B                   top 50     optional diversity
    └── exact-identifier channel                   NCT/patent/CAS/compound ONLY
                    ↓  dedupe, preserve channel provenance (never average scores)
        Qwen3-Reranker-8B, official prompt, 8K ctx  → top 40
                    ↓  (feature-flagged, gated)
        Diver-GroupRank-32B, random groupwise       → top 10
```

The one structural guarantee: **an uncapped union cannot have lower pre-rerank recall
than either constituent retriever.** Everything after the union is promoted only by
measured utility on independently judged data — see the gates below.

### Running it

```bash
# 0) THE BLOCKING STEP. The cascade needs vectors from the UNTOUCHED 0.6B, which do not
#    exist yet (~45 min, 600,738 docs). Without them every stage falls back to the OLD
#    fine-tuned-only pool -- the configuration that loses 0.089 R@10 on low-overlap.
python stages/p0_0_embed_channel.py \
    --model Qwen/Qwen3-Embedding-0.6B \
    --out-dir $DH2_ROOT/vectors/base06b

# 1) evaluate C0/C1/C2 on independently adjudicated queries only (~45-60 min/arm)
python stages/p1_4_eval_cascade.py \
    --vectors-base-06b $DH2_ROOT/vectors/base06b/doc_vectors.npy \
    --ids-base-06b     $DH2_ROOT/vectors/base06b/doc_ids.json \
    --vectors-8b       $DH2_ROOT/vectors/ft8b/doc_vectors.npy \
    --ids-8b           $DH2_ROOT/vectors/ft8b/doc_ids.json \
    --arms C0,C1,C2 --llm-judged-only
# -> reports/eval_cascade.md prints the gate table and a PROMOTE / DO NOT PROMOTE verdict

# 2) only if the gates pass: production, behind a flag (OFF is byte-identical to today)
export PYTHONPATH=/path/to/this/tree:$PYTHONPATH   # 07 imports dh2 ONLY when the flag is on
export DH_REGISTER_AWARE_CASCADE=1
export DH_ENABLE_GROUPRANK=0
export DH_CASCADE_VECTORS_BASE_06B=$DH2_ROOT/vectors/base06b/doc_vectors.npy
export DH_CASCADE_IDS_BASE_06B=$DH2_ROOT/vectors/base06b/doc_ids.json
python prod/07_retrieve_rank.py --query "Monoclonal antibody targeting PD-L1 for oncology"
# startup MUST print "[07] register-aware cascade ACTIVE". Anything else = old path.
```

**Hardware reality check.** Qwen3-Reranker-8B in BF16 is ~16 GB of weights: it does not
fit one RTX 3060 (12 GB), though `device_map="auto"` will shard it across two. GroupRank-32B
is ~64 GB and TP=4 — it does **not** fit on Drew at all and needs the rental box. C0/C1/C2
are feasible on Drew; C3 is not.

### Promotion gates (NOT exact-origin R@10)

`qrels_exact_origin` scores paraphrase retrieval — designated positives carry +0.100 more
query-word overlap than equally relevant alternatives — so it is deliberately not the
criterion. Promote the Qwen cascade only when **all** hold:

```
low-overlap utility does not regress
AND grade-3 MRR does not regress
AND overall graded nDCG@10 improves
AND no major source loses more than 2 percentage points
AND candidate Recall@100 >= every constituent retriever
```

GroupRank additionally needs: beats Qwen-only on independent judgments, low-overlap does
not regress, parse success ≥99%, latency acceptable. `p1_4` prints this table and a
PROMOTE / DO NOT PROMOTE verdict per arm.

### Anti-circularity

Score on three slices, never on teacher-consensus labels (96.6% grade-0, produced by the
same cross-encoders the cascade is built from — that measures agreement with itself):
1. LLM-adjudicated rows only (`--llm-judged-only`).
2. Newly surfaced documents, judged blind to channel and score.
3. ~25–50 blinded real-scout queries.

---

## Unattended run — three commands

All model/key config is via env vars (nothing hardcoded). On **Drew**:

```bash
# 0) one-time: put pipeline_2 on Drew (you rsync'd it here) and set the teacher key
export DH2_TEACHER_LLM_API_KEY='your-1min-ai-key'
export DH2_TEACHER_LLM_MODEL='claude-sonnet-4-6'

# 1) push code + data + models to the rented H100 (Vast alias e.g. vast_h100)
#    set PUSH_8B=1 to also send the 8B model + vectors (second teacher + pool source)
PUSH_8B=1 ./scripts/push_pipeline2_to_h100.sh vast_h100 /workspace

# 2) launch the unattended run on the box (detached; survives SSH drop)
ssh vast_h100 'cd /workspace/discovery_hub_pipeline_2 && source dh2_env.sh && \
    nohup bash scripts/run_pipeline2_h100.sh > $DH2_ROOT/logs/run.log 2>&1 &'

# 3) start the Drew-side watchdog: polls for DONE, auto-pulls results, destroys the box
#    (destruction happens from Drew where your Vast key lives — the box never holds it)
nohup ./scripts/watchdog.sh vast_h100 /workspace <vast_instance_id> > watchdog.log 2>&1 &
```

When the watchdog exits 0, results are in `/home/alex/discovery_hub/data/pipeline2/reports/`
and the winning model (if any) in `.../pipeline2/final_model/`. The box is destroyed.

### Resumability & safety
- Each stage writes a `.markers/<stage>.done`; re-running the orchestrator **skips
  completed stages** (safe after an interruption).
- Any stage failure writes `$DH2_ROOT/FAILED` and aborts; the watchdog pulls logs and
  **leaves the box up for debugging** (does not destroy).
- The LoRA merge **fails loud** if adapter tensors are missing — no arm can silently emit
  a random-init model (the pipeline_1 bug is fixed here).
- The LLM judge **caches** every verdict (`llm_cache.jsonl`) so a re-run never re-pays,
  and **masks** unparseable replies instead of crashing.

---

## Bakeoff arms (spec P1)

| Arm | Objective | Role |
|---|---|---|
| A `A_mnrl_control` | MNRL, single positive | locked control |
| B `B_mnrl_hardfilter` | MNRL, flagged negs removed | hard-filter control |
| C `C_gpl_marginmse` | GPL MarginMSE (teacher margins) | canonical soft-label baseline |
| D `D_listwise_kl` | Listwise KL + small contrastive | primary graded candidate |
| E `E_lsepair` | LSEPair (explicit multi-positive) | primary multi-positive candidate |
| F `F_rand1lh` | one random valid positive | simple multi-positive control |

Arms screen on the 0.6B backbone over a query subset; the winner (best exact R@10, nDCG
tiebreak) trains on 4B. Promotion gates (S5.2): **+0.02 R@10 exact AND +0.02 nDCG@10
utility, no MRR regression, no source R@10 regression > 0.02.**

---

## Not in this build (documented stubs — spec P2–P8)

Truncation audit / field-chunk indexing (P2), source query rewrites + HyDE + alias index
(P3), reranker A/B (P4), query-diversity re-mining (P5), conditional SPLADE (P6),
Matryoshka dims (P7), typed KG expansion + R-GCN (P8). Interfaces are stubbed in
`dh2/parallel_tracks_stubs.py` with pointers to the spec sections.

---

## Invariants carried from pipeline_1
- `docs.jsonl` line order == vector row order == index row order (never regenerate between
  embed and index). `manifest.doc_order_checksum()` fingerprints this.
- Query prefix `Instruct: {task}\nQuery: ` byte-identical between train and inference.
- `DH_EMBED_MODEL` must be set in every serving/eval shell; `validate.assert_serving_compat`
  fails fast on model/vector/dim mismatch.
- `unset HF_HUB_OFFLINE HF_HOME` on the box (Vast leaks these; they break the reranker).
