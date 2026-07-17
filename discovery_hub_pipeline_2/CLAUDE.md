# CLAUDE.md — DiscoveryHub RASC run: Drew → Vast H200 → Drew

You are operating on **Drew** (`alex@aiserver`). Your job is to push this pipeline to a
rented Vast box, run two stages there, monitor them, pull the results back, and report a
verdict. You are **not** here to train anything.

Read this whole file before running a command. Then start at **Phase 0**.

---

## 0. Prime directives

1. **DO NOT RETRAIN. DO NOT run `scripts/run_pipeline2_h100.sh`.** That is the old P0→P1
   training orchestrator. It ends by fine-tuning a model, which is the thing this work
   exists to *avoid*. If you find yourself running a bakeoff arm, you have gone wrong.
2. **DO NOT destroy the Vast box until `pull_pipeline2_results.sh` exits 0.** It exits
   non-zero on failure and exits **2** if `llm_cache.jsonl` is missing. Check `$?`. The
   previous version of that script printed "pull complete" and exited 0 having transferred
   nothing, right before the box was destroyed. Do not recreate that.
3. **Never commit, echo, paste, or transmit `dh2_env.sh` contents.** It holds the z.ai API
   key. It is `.gitignore`-worthy and must be scrubbed before anything is pulled back.
4. **`~/dh2_takehome/data/llm_cache.jsonl` is the single most expensive artifact in this
   project** — 52,709 LLM verdicts, real money. Back it up in Phase 0. Never overwrite it.
   Never change `DH2_TEACHER_LLM_MODEL` (the model string is hashed into every cache key;
   changing it silently invalidates all 52,709 verdicts).
5. **Verify, do not infer.** This project's own operational log is explicit: reading the
   source predicted outcomes correctly, but *every number extrapolated from a reading was
   wrong*. Run the check. `ls` the artifact. Don't report success from a log line.
6. **Stop and ask** before: renting/destroying anything, spending beyond the agreed budget,
   changing `DH_REGISTER_AWARE_CASCADE` in production, or if a gate FAILs.

---

## 1. What this is (context you need to make good calls)

DiscoveryHub matches pharma scouts to university tech (patents / trials / papers / SBIR).
On 2026-07-15 the fine-tuning bakeoff produced a null *and* a much better finding:

> Fine-tuning taught the model to match **words**, not meaning. The best arm gained
> **+0.165 R@10 on high-word-overlap queries and lost 0.089 on low-overlap ones**. The
> reported aggregate of +0.061 hid the trade. **76% of this corpus's judged true matches
> are low-overlap**, so the trade was bad. Root cause: synthetic queries are generated
> *from* their origin document, so they inherit its vocabulary — the training pairs don't
> contain the register gap, and neither does the benchmark.

The untouched stock **Qwen3-Embedding-0.6B**, which the pipeline never touched, remained
the **best low-overlap matcher of every arm evaluated** (0.2556 vs 0.1667).

So the plan is inference-only — the **Register-Aware Semantic Cascade (RASC)**:

```
scout query
  ├── untouched Qwen3-Embedding-0.6B  top 100   <- the semantic hedge
  ├── fine-tuned 8B                   top 100
  ├── fine-tuned 4B                   top 50
  └── exact-identifier channel (NCT/patent/CAS) — no global BM25
            ↓ union, dedupe, keep per-channel provenance, never average scores
      Qwen3-Reranker-8B (official prompt, 8K ctx, pharma instruction) → top 40
            ↓ (feature-flagged OFF, gated)
      Diver-GroupRank-32B → top 10
```

The guarantee it rests on: **an uncapped union cannot have lower recall than either
constituent retriever.** Everything after the union must be earned on measured utility.

**Never promote on exact-origin R@10.** It scores paraphrase retrieval (origin docs carry
+0.100 more query-word overlap than equally relevant alternatives). `p1_4` reports graded
utility bucketed by LOW/HIGH word overlap instead, and prints a PROMOTE verdict.

---

## 2. The machine

Rented Vast box (already selected — **do not shop for another**):

| | |
|---|---|
| GPU | **1× H200 NVL, 140 GB VRAM**, ~3.8 TB/s, CUDA 12.9 |
| CPU | AMD EPYC 9655, 48 vCPU |
| RAM | **242 GB** |
| Disk | ~1 TB NVMe (7.8 GB/s) |
| Region | Czechia |

**Single GPU ⇒ `DH2_GR_TP_SIZE=1`.** The default is 4 and will crash on one card.

VRAM budget (BF16 weights):

| | GB |
|---|---:|
| encoders (0.6B + 4B + 8B) | 25.2 |
| Qwen3-Reranker-8B (**shared across arms** — `p1_4` passes one instance) | 16.0 |
| **C2 total** | **41.2** |
| GroupRank-32B (C3 only) | 64.0 |
| C2 + C3 | 105.2 of 140 |

⚠️ If you ever run **C3**: vLLM reserves `gpu_memory_utilization × TOTAL` VRAM, not ×
*free*. The default 0.90 → 126 GB, leaving 14 GB for 41 GB of resident encoders → OOM.
Set `DH2_GR_GPU_UTIL=0.60` or run C3 in a separate process. **C3 is not required for this
run** — it is gated off. Prefer `--arms C0,C1,C2`.

---

## 3. Layout

| what | where |
|---|---|
| this tree | `~/discovery_hub_pipeline_3/` (Drew) — adjust if different, **verify first** |
| base package | `/home/alex/discovery_hub/` (`discovery_hub/` package + `07_retrieve_rank.py`) |
| corpus | `/home/alex/discovery_hub/data/normalized/docs.jsonl` (~600,738 docs) |
| models | `/home/alex/discovery_hub/models/` (`qwen3-dh-ft-4b`, `qwen3-dh-ft-8b`, `qwen3-dh-ft`) |
| prior artifacts | `~/dh2_takehome/data/` — `llm_cache.jsonl`, `qrels/`, `multi_positive_labels_v1.jsonl`, `reports/` |
| remote root | `/workspace` on `vast_h100` |

You may read anything under `/home/alex`. **Write only** to this tree, `$DH2_ROOT`, and
`/tmp`.

---

## 4. Phase 0 — preflight ON DREW (no rental yet)

Do all of this before spending a cent.

```bash
cd ~/discovery_hub_pipeline_3     # VERIFY this path; find it if wrong

# 0.1 the tests must pass locally (CPU-only, no GPU, ~1 min)
python3 tests/test_dh2_smoke.py
python3 tests/test_losses_bugfix.py       # needs torch; skip if absent, note it
python3 tests/test_07_cascade_flags.py

# 0.2 BACK UP THE EXPENSIVE ARTIFACT before anything else touches it
cp ~/dh2_takehome/data/llm_cache.jsonl \
   ~/dh2_takehome/data/llm_cache.jsonl.bak.$(date +%F)
wc -l ~/dh2_takehome/data/llm_cache.jsonl    # expect ~52,709

# 0.3 inventory what actually exists. Report a TABLE; do not assume.
ls -la /home/alex/discovery_hub/models/
ls -la /home/alex/discovery_hub/data/normalized/docs.jsonl
ls -la /home/alex/discovery_hub/data/embeddings*/ 2>/dev/null
ls -la ~/dh2_takehome/data/qrels/
nvidia-smi     # what does Drew itself have?
ssh vast_h100 'nvidia-smi; free -g; df -h /workspace'
```

**Report to the user before Phase 1**, as a table:
- which models exist on Drew (4B? 8B? 0.6B?)
- which doc-vector sets exist (4B production? 8B? base-0.6B — **expected ABSENT**)
- whether `qrels_scout_utility_v1.jsonl` + `multi_positive_labels_v1.jsonl` are present
- H200 confirmed, free RAM, free disk
- **anything missing that blocks Phase 2**

If `qwen3-dh-ft-8b` is missing on Drew, say so — the 8B channel is optional; C2 can run
with the semantic hedge + deployed 4B, but say it out loud rather than silently degrading.

---

## 5. Phase 1 — push to the box

```bash
export DH2_TEACHER_LLM_API_KEY='<ask the user; never echo it>'
export DH2_TEACHER_LLM_MODEL='glm-4.6'                       # DO NOT CHANGE (cache keys)
export DH2_TEACHER_LLM_BASE_URL='https://api.z.ai/api/paas/v4'

PUSH_8B=1 ./scripts/push_pipeline2_to_h100.sh vast_h100 /workspace
```

This rsyncs code + data + models and writes `/workspace/discovery_hub_pipeline_2/dh2_env.sh`
on the box. That file **contains the key**. Never cat it into a log or a message.

The push script writes the correct `glm-4.6` + z.ai base URL defaults. If you see
`claude-sonnet-4-6` anywhere, that is the dead 1min.ai provider — **stop and report it**.

Verify the push landed (the artifact is the proof, not the log):

```bash
ssh vast_h100 'ls -la /workspace/discovery_hub_pipeline_2/ && \
               ls -la /workspace/models/ && \
               du -sh /workspace/dh_data/'
```

---

## 6. Phase 2 — embed the semantic hedge (~45 min) ⚠️ BLOCKING

**Nothing works without this.** The cascade needs doc vectors from the **untouched**
0.6B. They do not exist. Without them, `p0_1`/`p1_4`/`07` all warn and fall back to the
**old fine-tuned-only pool** — the exact configuration measured to lose 0.089 R@10 on
low-overlap queries. A cascade without the hedge ships the regression under a new name.

**tmux rules (learned the hard way — do not improvise):**
- `tmux new-window` does **NOT** inherit your shell env. Put `cd` + `source dh2_env.sh`
  **inside a wrapper script**; never pass a bare command. (This cost 40 minutes once.)
- Verify the wrapper exists (`ls -la`) **before** launching. A missing file makes the
  window open and close instantly, which looks exactly like a crash.
- A window that vanishes from the status bar **exited**; it wasn't killed. Read the log.
- `tmux capture-pane` looks empty because of tqdm. Pipe through `tr '\r' '\n'`.

```bash
ssh vast_h100 'cat > /workspace/run_embed.sh' <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/discovery_hub_pipeline_2
source dh2_env.sh
unset HF_HUB_OFFLINE HF_HOME          # Vast leaks these; they break HF downloads
export DH2_GR_TP_SIZE=1               # single H200
mkdir -p "$DH2_ROOT/logs"
$PYTHON stages/p0_0_embed_channel.py \
    --model Qwen/Qwen3-Embedding-0.6B \
    --out-dir "$DH2_ROOT/vectors/base06b" \
    --batch-size 256 \
    > "$DH2_ROOT/logs/p0_0.log" 2>&1
echo "EXIT=$?" >> "$DH2_ROOT/logs/p0_0.log"
EOF
ssh vast_h100 'chmod +x /workspace/run_embed.sh && ls -la /workspace/run_embed.sh'
ssh vast_h100 'tmux new-session -d -s embed "/workspace/run_embed.sh"'
```

**Monitor** (poll every ~5 min; the artifact is the honest progress meter):

```bash
ssh vast_h100 'tail -5 /workspace/dh_data/pipeline2/logs/p0_0.log | tr "\r" "\n"'
ssh vast_h100 'ls -la /workspace/dh_data/pipeline2/vectors/base06b/ 2>/dev/null'
ssh vast_h100 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

**Done when** `doc_vectors.npy` exists and `p0_0` printed its shape. Expect
`600738 x 1024` (~2.5 GB) and `doc_order_checksum`. `p0_0` self-checks that vectors are
unit-norm and that rows == ids; it refuses to clobber without `--force`.

If it OOMs, drop `--batch-size` to 128 or 64. 140 GB should be ample for a 0.6B.

---

## 7. Phase 3 — evaluate the cascade (~45–60 min per arm)

```bash
ssh vast_h100 'cat > /workspace/run_eval.sh' <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
cd /workspace/discovery_hub_pipeline_2
source dh2_env.sh
unset HF_HUB_OFFLINE HF_HOME
export DH2_GR_TP_SIZE=1
V="$DH2_ROOT/vectors"
$PYTHON stages/p1_4_eval_cascade.py \
    --vectors-base-06b "$V/base06b/doc_vectors.npy" \
    --ids-base-06b     "$V/base06b/doc_ids.json" \
    --vectors-8b       "${DH2_VECTORS_8B:-}" \
    --ids-8b           "${DH2_IDS_8B:-}" \
    --arms C0,C1,C2 \
    --llm-judged-only \
    --name cascade \
    > "$DH2_ROOT/logs/p1_4.log" 2>&1
echo "EXIT=$?" >> "$DH2_ROOT/logs/p1_4.log"
EOF
ssh vast_h100 'chmod +x /workspace/run_eval.sh && ls -la /workspace/run_eval.sh'
ssh vast_h100 'tmux new-session -d -s eval "/workspace/run_eval.sh"'
```

`--llm-judged-only` is **not optional**. It restricts scoring to independently
LLM-adjudicated queries. Teacher-consensus labels are 96.6% grade-0 and were produced by
the same cross-encoders the cascade is built from — scoring against them measures the
system's agreement with itself.

Watch for these lines in the log and **report them**:
- `[p1_4] channel base_06b : ready` ← if `ABSENT`, Phase 2 didn't land; **stop**
- `[p1_4] WARNING: no untouched-0.6B vectors` ← **stop**, you are evaluating the old pool
- per-arm `nDCG@10 overall=… LOW=… HIGH=…`

---

## 7c. Phase 3c — settle the premise (backbone-controlled) + locate the bottleneck

The 2026-07-16 run produced a corrected **DO NOT PROMOTE** for C2 (low-overlap −0.110,
overall nDCG −0.017 vs C1) and, incidentally, a result that **inverts this project's
founding premise**: on the 853 independently-adjudicated queries the *stock* 0.6B was the
**worst** low-overlap channel (0.1620) while the FT-8B was the best (0.3578). The bakeoff
claim ("fine-tuning costs low-overlap recall") came from a **different eval** — 7,545
*synthetic* queries with curated labels — so the two are not comparable.

**The bakeoff held the backbone fixed at 0.6B. So the replication must too.** A
0.6B-vs-8B comparison confounds fine-tuning with model size and answers nothing.

| arm | what it is | reads |
|---|---|---|
| `C0` | stock 0.6B, dense | LOW 0.1620 (have) |
| `C0ft` | **fine-tuned 0.6B, dense** — same backbone as C0 | the replication |
| `C1` | FT-8B, dense | LOW 0.3578 (have) |
| `C1b` | FT-8B top-100 → **shared Qwen rerank** | union vs reranker |
| `C2` | full cascade | (have) |

Read `C0ft` against `C0` **only** — same backbone, same eval set, same metric:
- **`C0ft` LOW < 0.1620** → the premise replicates on independent judgments. Fine-tuning
  really does cost low-overlap. RASC's foundation holds.
- **`C0ft` LOW > 0.1620** → the premise was a **synthetic-eval artifact**. Fine-tuning helps
  even at 0.6B once a real judge scores it, and **"do not retrain" needs revisiting before
  anyone pitches it.**

`C1b` isolates the stage: **C1b vs C1** = what the *reranker* did to a fixed candidate set;
**C2 vs C1b** = what the *union* added on top. The `oracle nDCG@10` / `headroom` columns
(`oracle_utility` over each arm's own pool) bound what any reranker could achieve — large
headroom means the docs are already retrieved and the ranking stage is the bottleneck.

`C0b` (untouched **8B**) answers a *different* question — size or fine-tuning? — and only
matters once you know whether the premise replicates. **Defer it.**

```bash
# 1. embed the FINE-TUNED 0.6B (~45 min). Doc side takes NO instruction prefix.
$PYTHON stages/p0_0_embed_channel.py --model "$DH2_MODEL_06B" \
        --out-dir "$DH2_ROOT/vectors/ft06b" --batch-size 256

# 2. one run, all arms (~30 min; ONE shared Qwen reranker serves C1b and C2)
$PYTHON stages/p1_4_eval_cascade.py \
    --vectors-base-06b "$DH2_VECTORS_BASE_06B" --ids-base-06b "$DH2_IDS_BASE_06B" \
    --vectors-ft-06b   "$DH2_ROOT/vectors/ft06b/doc_vectors.npy" \
    --ids-ft-06b       "$DH2_ROOT/vectors/ft06b/doc_ids.json" \
    --vectors-8b "$DH2_VECTORS_8B" --ids-8b "$DH2_IDS_8B" \
    --arms C0,C0ft,C1,C1b,C2 --llm-judged-only --name settle
```

`C0ft` is deliberately **not** in `retrievers`, so it never enters C2's union and C2 stays
comparable to the prior run. Still `--llm-judged-only`: teacher-consensus labels are 96.6%
grade-0 and were produced by the cross-encoders the cascade is built from.

---

## 8. Phase 4 — pull results back to Drew

```bash
cd ~/discovery_hub_pipeline_3
./scripts/pull_pipeline2_results.sh vast_h100 /workspace ~/dh2_takehome/data
echo "PULL EXIT: $?"     # <-- MUST be 0. 2 = llm_cache.jsonl missing. Anything else = failure.
```

The script scrubs the API key from `dh2_env.sh` on the box, tars a named manifest
(**including `adjudication/` and `teacher_scores/`**, the two expensive directories the
old script silently omitted), verifies size + integrity after transfer, and fails loudly.

**Only if it exits 0** may you propose destroying the box — and destruction is the user's
call, from Drew, where the Vast key lives. Never put the Vast key on the box.

Then read the verdict:

```bash
cat ~/dh2_takehome/data/reports/eval_cascade.md
```

---

## 9. Phase 5 — report

Give the user a short table, not a narrative:

| arm | nDCG@10 | LOW nDCG | HIGH nDCG | Success@10 | grade-3 MRR | cand R@100 | p95 ms |

Then the gate block `p1_4` printed, verbatim, and its **PROMOTE / DO NOT PROMOTE**
verdict. Gates (all must hold for C2 vs its best constituent):

```
low-overlap utility does not regress
AND grade-3 MRR does not regress
AND overall graded nDCG@10 improves
AND no source loses more than 2 percentage points
AND candidate Recall@100 >= every constituent retriever
```

**Do not editorialize the verdict.** If C2 fails a gate, say it failed and stop. Do not
suggest flipping `DH_REGISTER_AWARE_CASCADE=1` in production. A failed gate on this eval
is a *useful* result: the register-gap finding stands on its own and is the stronger story
either way.

If C2 **passes**, tell the user the production flip is a separate, deliberate step:

```bash
export PYTHONPATH=/home/alex/discovery_hub_pipeline_3:$PYTHONPATH   # 07 imports dh2
export DH_REGISTER_AWARE_CASCADE=1
export DH_ENABLE_GROUPRANK=0
export DH_CASCADE_VECTORS_BASE_06B=<path>/doc_vectors.npy
export DH_CASCADE_IDS_BASE_06B=<path>/doc_ids.json
```
Startup **must** print `[07] register-aware cascade ACTIVE`. Anything else means it
silently fell back to the old path. `DH_REGISTER_AWARE_CASCADE=0` is byte-identical to
today's behavior (`tests/test_07_cascade_flags.py` monkeypatches `_build_cascade` to raise,
proving the flag-off path never reaches it).

---

## 10. Failure playbook

| symptom | do this |
|---|---|
| tmux window vanished instantly | It exited. `cat` the log. Check the wrapper exists. |
| `capture-pane` looks empty | tqdm. `tr '\r' '\n'`. |
| `channel base_06b: ABSENT` | Phase 2 didn't produce vectors. **Do not proceed.** |
| CUDA OOM in p1_4 | Confirm one shared reranker (16 GB, not 32). Lower `--batch-size`. Don't run C3. |
| HF download fails on the box | `unset HF_HUB_OFFLINE HF_HOME`. Vast leaks them. |
| vLLM OOM (C3 only) | `DH2_GR_GPU_UTIL=0.60`, `DH2_GR_TP_SIZE=1`, or separate process. |
| pull exits non-zero | **Leave the box up.** Report. Do not destroy. |
| a stage fails | `$DH2_ROOT/FAILED` is written. Pull logs. Leave the box up. |
| `scp`/`ssh` port confusion | `scp` takes `-P`, `ssh` takes `-p`. |

**Rough budget:** Phase 2 ~45 min + Phase 3 ~1–3 h ⇒ ~2–4 h of H200 rental. If you are
about to exceed ~6 h, stop and ask.

---

## 11. Things you must never do

- Run `run_pipeline2_h100.sh`, `p1_2_run_bakeoff.py`, or any training.
- Change `DH2_TEACHER_LLM_MODEL` (invalidates 52,709 cached verdicts).
- Set `DH2_TEACHER_LLM_CACHE_MODEL_ID` (makes one judge inherit another's verdicts — a
  scientific claim, not a config tweak).
- Destroy the box before a clean pull.
- Print, commit, or transmit the contents of `dh2_env.sh`.
- Overwrite `llm_cache.jsonl`, `teacher_scores_v1.jsonl`, or anything in `~/dh2_takehome/`
  other than by an explicit, verified pull.
- Report success from a log line. `ls` the artifact.
