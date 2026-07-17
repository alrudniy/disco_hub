# DiscoveryHub pipeline_2 — Retrieval-Quality R&D (P0 + P1 critical path)

An **additive** pipeline that implements the highest-leverage part of the OpenAI tech
spec: the **P0 (pooled multi-positive evaluation)** and **P1 (multi-teacher, multi-positive
distillation bakeoff)** critical path. It reuses the existing `discovery_hub` package
(one source of truth for corpus/serving config) and runs the compute-heavy work on a
rented Vast H100, then pulls results back to Drew — **unattended**.

This is **post-demo R&D**: a full run (candidate pooling → cross-encoder scoring → LLM
adjudication → objective bakeoff → train winner → re-embed/eval) is many GPU-hours and
cannot finish before the July 17 demo. Build now, run after.

> **Teacher design:** the primary relevance scores come from **two cross-encoders**
> (BGE-reranker-v2-m3 + Qwen3-Reranker-8B on the H100). **claude-sonnet-4-6 (via 1min.ai)**
> is the **gray-zone judge** for teacher disagreements, and the external LLM teacher where
> the spec calls for one. Because human calibration is replaced by the LLM, all outputs are
> framed as **"LLM-adjudicated, not human-calibrated"** (spec S2.4 / risk register).

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
│   ├── llm_client.py          # claude-sonnet-4-6 via 1min.ai (retry/cache/mask)
│   ├── retriever2.py          # load ANY (model, vectors, ids) for candidate pooling
│   ├── candidate_pool.py      # P0/P1 pooling: 4B top + 4B deep-window + 8B top
│   ├── teachers.py            # BgeTeacher + QwenTeacher (yes-token prob readout)
│   ├── teacher_merge.py       # agreement → gray-zone LLM → graded label / mask
│   ├── eval_pool_build.py     # exact + graded qrels; held-out split; family-collapse
│   ├── graded_metrics.py      # nDCG@10, utility-Recall, grade-3 Recall, bootstrap CIs
│   ├── losses.py              # MarginMSE, listwise-KL, LSEPair (+ torch modules)
│   ├── train_bakeoff.py       # arms A–F; screen on 0.6B; train winner on 4B (safe merge)
│   ├── validate.py            # LoRA-merge FIX + startup dim/prefix/index checks (S10)
│   ├── manifest.py            # §9.2 version metadata + doc-order checksum
│   └── parallel_tracks_stubs.py  # P2–P8 interface stubs (documented, not implemented)
├── stages/                    # thin runnables (argparse → dh2/*)
│   ├── p0_1_build_candidate_pool.py
│   ├── p0_2_score_teachers.py
│   ├── p0_3_llm_adjudicate.py
│   ├── p0_4_build_eval_qrels.py
│   ├── p1_1_build_train_labels.py
│   ├── p1_2_run_bakeoff.py
│   └── p1_3_eval_and_compare.py
├── scripts/
│   ├── push_pipeline2_to_h100.sh   # Drew → H100 (code+data+models+env)
│   ├── run_pipeline2_h100.sh       # THE unattended orchestrator (writes DONE sentinel)
│   ├── pull_pipeline2_results.sh   # H100 → Drew
│   └── watchdog.sh                 # Drew-side: poll DONE → pull → destroy box
├── tests/test_dh2_smoke.py    # mini-corpus end-to-end (no GPU/net/torch)
├── requirements2.txt
└── README.md
```

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
