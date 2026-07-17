# Discovery Hub — the multi-agent layer

This README describes **what was built**, against `multiagent_build_spec.md`, and — with equal
weight — what was found to be wrong on the way. Read [`agents/README.md`](agents/README.md) for the
agent-by-agent map and the honest scorecard.

---

## 0. Why this exists

The pitch deck claimed a *"Multi-agent pipeline (retrieval, expertise-gap, reranking, policy/safety)"*
and *"multi-agent verification"*. Before this build, the serving path was:

```
query → FAISS dense (+BM25) → BGE cross-encoder rerank → RAG answer + confidence gate
```

Two of four claimed agents, no orchestration, no verification. `08_multiagent_rag.py` was named
"multiagent" and was a single-pass RAG — the word was in the filename, not the architecture.
`multiagent_build_spec.md` was written to close that gap in code rather than in tense. This is the
result of executing it.

## 1. What now exists

```
                          ┌─────────────────┐
   scout query ─────────► │  Orchestrator   │   route → fan-out → aggregate → gate → trace
                          └────────┬────────┘
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │  Retrieval   │    │ Expertise-   │    │   Policy /   │
      │  (wraps 07)  │    │  gap agent   │    │ safety agent │
      └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
              │                   │                   │
              ▼                   │                   │
      ┌──────────────┐            │                   │
      │  Reranking   │            │                   │
      │  (inside 07) │            │                   │
      └───────┬──────┘            │                   │
              └───────────┬───────┴───────────────────┘
                          ▼
                 ┌──────────────────┐
                 │  Synthesis (RAG) │   ← 08's strict-RAG step, re-pointed at z.ai/glm-4.6
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │  Verifier agent  │   ← evidence-gating (NOT "verification")
                 └────────┬─────────┘
                          ▼
            answer + evidence + gaps + flags + verdict + trace
```

| file | what it is |
|---|---|
| `agents/base.py` | `AgentResult` / `Agent` protocol / timing / error isolation. Three invariants: abstention is success; one agent failing never kills the run; confidences are never averaged across agents. |
| `agents/graph_index.py` | Offline builder + loader for a 286 MB CSR view of the merged graph (1,489,785 nodes / 4,520,029 edges). Loads in ~1 s. |
| `agents/retrieval.py` | Wraps `07_retrieve_rank.py::Retriever`. No retrieval logic lives here. |
| `agents/expertise_gap.py` | **The differentiator.** Graph traversal: portfolio → capability footprint → targets far from it → who owns them. |
| `agents/synthesis.py` | 08's strict-RAG answer step via `llm.py`. |
| `agents/verifier.py` | Evidence-gating. Adversarial framing, deliberately divergent from synthesis's. |
| `agents/policy.py` | Answer-safety, spec §4.5**(a)** only. |
| `agents/orchestrator.py` | Route → fan-out → aggregate → gate → trace. |
| `agents/llm.py` | Optional glm-4.6 client. No key → `.available` is False → every agent falls back **and says so**. |
| `agents/LLM_CONTRACT.md` | The **verified** glm-4.6 call contract. Read before touching `llm.py`. |
| `demo/run_demo.py` | The three scripted queries. Prints the trace table. |
| `demo/env.sh` | The corrected env block. The spec's own setup block does not work — see §3. |
| `demo/validate_portfolios.py` | Hand-validation of 5 org portfolios, mandated by spec §4.3. |

## 2. Run it

```bash
source demo/env.sh                        # corrected paths/dims; sources .env if present
venv/bin/python demo/run_demo.py          # the three scripted queries
venv/bin/python demo/run_demo.py --mock   # plumbing only, no real index, no GPU
venv/bin/python -m pytest tests/ -q
```

The LLM key lives in `.env` (gitignored, mode 600). Without it the system **still runs**, on
deterministic fallbacks that label themselves as such. A demo that silently pretends an LLM ran is
the failure mode this repo exists to avoid.

Memory: a real Retriever load is ~20 GB on a 30 GB box. **Do not construct two.**

## 3. What the build found

The spec is a good document, and it is **wrong about the data in five places**. Each was verified by
direct inspection before any code was written against it. They are recorded here because the next
person will otherwise trust the prose.

| # | Spec says | Reality (verified) |
|---|---|---|
| 1 | `expert --affiliated_with--> org` — "the `expert` node type exists; use it" | **That path has zero edges.** Experts reach orgs only via `expert ←investigated_by— tech —assigned_to→ org` |
| 2 | `org --assigned_to--> tech` | Reversed: `technology --assigned_to--> organization`. Portfolios need a reverse index |
| 3 | (silent) | Org identity is **fragmented**; a single-node portfolio is wrong — see below |
| 4 | `export DH_GRAPH_DIR` / `DH_USE_KEYWORD` | `config.py` read **neither**. One `DATA_ROOT` cannot address both the graph (`data_merged/`) and the embeddings (`data/`) |
| 5 | (silent) | Default config **crashes**: it names a 1024-dim model, but `doc_vectors.npy` and `faiss.index` are **2560-dim** (the fine-tuned local 4B model) |

Also: the edge key is `rel`, not `relation`.

**§4.5(b) is not buildable, and that is now settled.** The spec said to verify `priority_date` exists
before designing around it. It does not — `docs.jsonl` has no `priority_date`, no filing/grant dates,
and `extra{}` is empty corpus-wide. Patent expiry cannot be computed. Policy ships **scope (a) only**,
and deliberately does not fire on IP queries: a run that found zero flags would render as
"policy: fired, clean", which reads as *the IP was checked and it's fine*. It wasn't.

### Org fragmentation is load-bearing, not a footnote

`organization:bristol myers squibb` has **826** assigned technologies. Canonicalizing corporate
suffixes finds five spelling variants of the same company:

```
'Bristol-Myers Squibb', 'BRISTOL MYERS SQUIBB CO', 'BRISTOL=MYERS SQUIBB COMPANY',
'Bristol_Myers Squibb Company', 'BRISTOL MYERS SQUIBB INC'
```

— note `BRISTOL=MYERS` and `Bristol_Myers`, real damage in the source data. Merged, the portfolio is
**3,155** technologies. The naive single-node lookup the spec describes would miss **74%** of the
portfolio and confidently report gaps in areas BMS already covers — wrong in the one direction a
pharma partner would catch instantly.

Subsidiaries (`Juno Therapeutics, Inc., a Bristol-Myers Squibb Company`, Karuna, Adnexus) are
**deliberately not merged**: they are distinct legal entities and folding them in would inflate the
portfolio. Yale and Mayo fragment by word order (`yale university` / `univ yale` / `yale univ`), which
suffix-stripping **cannot** fix. That needs entity resolution, not a threshold.

### A pre-existing bug in `entity_link.py`, found and fixed

The alias rule promoted any globally-unique token of a multi-word label. Uniqueness is precisely the
wrong criterion: in ~926k surfaces, a distinctive company name is often *shared* while an odd word
appears exactly once. It produced real links to real nodes:

```
"CXCR4 gene therapy approach"          → organization:science approach
"Oral tablet formulation for CML"      → organization:hot album tansansen tablet inc
"Drugs targeting glutamate signaling"  → organization:cell signaling technology inc
```

The rule is now structural: a token is promoted only if stripping trailing legal suffixes leaves
exactly that token — i.e. it *is* the entity's whole name (`Genentech Inc` → `genentech`), never a
fragment. A Zipf-frequency filter rejects ordinary English words. **Known, deliberate cost:** `"lilly"`
no longer aliases Eli Lilly (two tokens, no legal suffix); the full surface still links.
`tests/test_hybrid.py` was updated to pin the new rule — it previously asserted the refuted behavior.

This is plausibly a root cause of the graph's measured **−0.0243 nDCG** retrieval penalty: the graph
may not be a bad channel so much as one that was anchoring on garbage. **That is a hypothesis, not a
result.** Confirming it means re-running stage 10. Do not put it in a deck until someone does.

## 4. What the demo actually does (measured, on the H200)

Run end to end on `vast_h100` (H200 NVL, 143 GB VRAM, 2 TB RAM), which mirrors Drew exactly:
same 603,369 docs, same (603369, 2560) `doc_vectors`, same 1,489,785-node graph, same
`faiss.index` byte size. ~135 s for all three queries, LLM live. Real output, not a script:

| query | trace | outcome |
|---|---|---|
| 1. the hit | retrieval 1.00, **expertise-gap abstained**, **policy abstained**, verifier **pass** | answer delivered, cited |
| 2. the differentiator | retrieval 0.40, **expertise-gap fired 0.85**, verifier **pass** | 90 targets analysed, gaps + licensing leads |
| 3. the refusal | retrieval 0.98, **policy blocked (dosing)**, verifier **fail** | **withheld** → evidence-only |

Query 2 returns what the tool is for — B7-family technologies BMS does not own, each with the
graph path to whoever does:

| gap | who covers it |
|---|---|
| `US7238360B2` Alteration of cell membrane with B7 | Univ Louisville Res Found — inventor Shirwan Haval |
| `US11547739B2` Peptides related to ICOS signaling | La Jolla Institute — Crotty Shane, Altman Amnon |
| `US10716838B2` Anti-CD277 antibodies | INSERM / CNRS / Univ Aix Marseille |

### The reranker eats the semantic bridge (the most useful result here)

The pitch's core claim is that this finds what a keyword search cannot. Measured, on the
real corpus, the dense encoder **does** — and the cross-encoder then **throws it away**.

Query: `HER2 targeted therapy for breast cancer`. 253 documents in the corpus say "ErbB2"
and never "HER2" in any spelling; a HER2 keyword search cannot return one of them. Dense
cosine ranks `uspto:US11903948B2` "Anti-ErbB2 antibody-drug conjugate" **9th of 603,369**
(0.561). `her2` and `erbb2` share no characters — that is real synonym knowledge.

Then the BGE cross-encoder reranks the 50-candidate pool:

| of the 50 reranked candidates | count | positions after rerank |
|---|---:|---|
| literally contain "HER2" | 45 | 1–45 |
| do **not** (the bridged docs) | 5 | **46, 47, 48, 49, 50** |

Exactly the last five. The ErbB2 patent lands **49th**. The reranked top-10 is **100%**
literal-HER2.

That partition is a correlation, so it was probed causally. Take the **rank-1** document,
"Antigen-binding constructs targeting HER2", and rename HER2 → ErbB2 — the same protein,
every other byte identical, still 331 tokens of on-topic antibody engineering:

| intervention | Δ logit | Δ rank |
|---|---:|---:|
| rank-1 doc, HER2 → ErbB2 (6 subs) | **−5.4711** | 1 → **38** |
| the ErbB2 patent, ErbB2 → HER2 (1 sub) | **+1.7618** | 49 → **38** |

**The claim, and only this claim: the reranker carries a lexical dependence large enough,
on this query, to bury a semantically identical document 37 places.**

Three things it is **not**, each ruled out by a number above:

- **Not a keyword filter.** The 45 literal docs span ~100× in score (0.0058–0.6296); the
  model discriminates hard among them, and it disagrees with dense in both directions
  (rerank #1 was dense #46; dense #1 fell to rerank #29).
- **Not a constant lexical bonus.** −5.47 vs +1.76 is a 3.1× disagreement. Whether the
  shape is dose-saturating (the rank-1 doc had 6 mentions, the bridge 1) or a context
  interaction is **not yet established** — see `analysis/rerank_lexical_dependence/`.
- **Not the whole story.** Giving the bridge document the magic token buys +1.76 logits —
  to rank 38, **not into the top 10**. Surface form is a large part of what buries it and
  not all of it; its Markush abstract genuinely says little.

Ruled out as explanations: truncation (0 of 50 pairs truncated; both stages read the same
538-char `embedding_text`), degenerate scores (5 distinct bridged values, no 0/NaN/floor),
non-determinism (bitwise identical across runs), and a length confound (bridged median 201
tokens vs literal 213, well inside the literal range).

This is not a tuning problem, it is a locating one: the register-gap failure
`pipeline_3`'s handoff attributes to fine-tuning is *also* present, undiluted, in the
off-the-shelf reranker that runs after it. Any claim about crossing nomenclature has to
survive `_rerank()`, and today nothing does. Cheapest next probe: re-run stage 10 with
reranking disabled and compare — if dense-only recall beats retrieve-then-rerank on
cross-register queries, the reranker is a liability on exactly the queries this product
exists for.

### Two things the spec promised that did NOT happen

**The hit does not hit.** Spec §5 says retrieval surfaces `US6803192B1` ("B7-H1, a novel
immunoregulatory molecule") because B7-H1 is the older name for PD-L1, and that "no keyword
search finds this". The document IS in the corpus and IS in the graph. **It ranks 7,585 of
603,369**, dense cosine **0.2342**, against a top-10 at ~0.60. The encoder does not bridge the
B7-H1 → PD-L1 synonym at all; it scores them as near-unrelated. Do not tell this story in a
room — it is checkable in ten minutes by anyone who asks.

That is not one unlucky pair. Sweeping **28 synonym pairs**, each controlled so the queried
term is absent from the candidate documents in *every* spelling (so a keyword search provably
cannot return them), only **HER2/ErbB2** put a bridged document in the dense top-10. Arbitrary
identifiers fail completely: `Opdivo` 56,968 · `Ozempic` 137,871 · `CD274` **287,301** of
603,369. The encoder bridges names that share morphology or descriptive structure
(`acetylsalicylic acid`→aspirin, `HMG-CoA reductase inhibitor`→statin, `ErbB2`→HER2) and is
blind to codes. And the one bridge that works is then destroyed by the reranker, above.

A caution about method, because it nearly went the other way: the first pass of that sweep
used a plain substring exclusion and reported PD-1/CD279 at rank 3 and B7-H3/CD276 at rank 17
as bridges. Both were lexical leaks — the documents spell the modern term without a hyphen
("Anti-**PD1** antibodies", "Anti-CD276 antibodies **(B7H3)**"). Under a separator-blind
control they collapse to **2,434** and **5,466**. A synonym experiment that does not normalize
punctuation measures its own tokenizer.

**The money shot's cast is different.** Spec §5 predicts Chen Lieping's patents, assigned to
Mayo/Yale, as the fillers. The real fillers are Louisville, La Jolla and INSERM (above). Chen
Lieping's 52 patents and his Mayo/Yale/BMS affiliations are all real in the graph — they are
simply not what this query's field retrieval returns.

## 5. What is measured, and what is not

**Verified directly during this build** (re-checkable in minutes):

- Graph: 1,489,785 nodes / 4,520,029 edges. `organization:bristol myers squibb` → **826** assigned
  technologies (**3,155** across 5 canonical variants). `inventor:chen lieping` → **52** `invented_by`
  technologies, affiliated with Mayo Foundation, Yale, Johns Hopkins and BMS. The demo's licensing-lead
  traversal is backed by real edges.
- Edge type-pairs, exhaustively, by full scan of all 4.52M edges (§3 above).
- glm-4.6 call contract, by live calls — including that it is a **reasoning model** whose reasoning
  tokens bill against `max_tokens` and silently return `content: ""` when they exhaust it. See
  `agents/LLM_CONTRACT.md`.
- Test suite: **284 passing**. One failure, `tests/test_pipeline_smoke.py`, is **pre-existing and
  unrelated** — it shells out to `01_download_data.py`, renamed to `01_download_data_v2.py` before
  this work. Confirmed never present in `HEAD`.

**Reported by the build, not independently re-verified** — treat as claims until someone re-runs them:
recall@100 0.7736 (dense) vs 0.6798 (RRF+keyword) on 123 adjudicated queries; the graph's zero unique
relevant docs across 62 firing queries; the policy false-positive probe over 20,000 abstracts (4 fires,
0.02%); expert→org reachability 13.4%; 53.6% of org nodes owning exactly one technology.

**Unmeasured, and must be said out loud:**

- **The expertise-gap agent has no eval.** No judged set of gap analyses exists, so there is no
  accuracy number and will not be one until someone builds the labels. `payload["unmeasured"]` is
  `True` on every result. It is a **capability demo, not a measured result.** Gaps are leads to check.
- **The verifier's effectiveness is unmeasured.** It reduces the rate at which ungrounded claims reach
  the user. By how much is unknown. Do not put a number on it.
- **Every threshold is hand-set**, not tuned: gap cosine 0.40, support 0.65, min portfolio 5. No
  labelled set exists to tune against; each one's comment says so.
- **No confidence here is calibrated**, and they are never averaged across agents — the scales are
  unrelated and their mean would be a fourth number meaning nothing.

## 6. What not to claim, even now

- **Not "verified" — "evidence-gated."** The verifier is an LLM checking an LLM against retrieved
  text. It answers *"does this retrieved passage state this sentence?"*, never *"is this sentence
  true?"*. A claim can be true and land `unsupported`; a claim can be false and land `supported`.
  Both are correct behavior.
- **No accuracy number for expertise-gap.** There is no benchmark. Saying otherwise invents one.
- **The graph's *retrieval* contribution is measured and negative.** Claiming the graph powers
  expertise-gap is defensible. Claiming it improves *search* contradicts our own data.
- **Fan-out is sequential, and the trace says so.** Spec §4.2 says retrieval and expertise-gap run
  concurrently; its own code sketch feeds retrieval's candidates *into* expertise-gap, which makes
  them dependent. Both cannot be true. Wrapping a real dependency in a `ThreadPoolExecutor` would put
  fake parallelism in the trace — the exact category of thing this spec exists to prevent.

The honest sentence for the deck is in `multiagent_build_spec.md` §0, and it survives the follow-up
question. The current one does not.
