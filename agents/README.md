# `agents/` — the multi-agent layer

An honest map. If you are here to check whether the deck's claims survive a
technical read, start with [What is measured and what is not](#what-is-measured-and-what-is-not)
and the [scorecard](#scorecard-spec-section-7-updated).

Run it:

```bash
source demo/env.sh                          # the CORRECTED env — see below
venv/bin/python demo/run_demo.py            # the three scripted queries
venv/bin/python demo/run_demo.py --mock     # plumbing only, no real index
```

---

## The pipeline

```
query
  │
  ├─► retrieval        ALWAYS runs.        07's Retriever, wrapped. dense (+BM25 +graph, both off)
  │        │
  │        ▼ candidates
  ├─► expertise-gap    ~45% of queries.    graph traversal: portfolio → gaps → who fills them
  │        │            routes on: does the query name an org?
  │        ▼ gaps
  ├─► synthesis        cited answer, strict RAG, from retrieval's candidates only
  │        │
  │        ▼ answer
  ├─► policy           routes on: does the exchange make a clinical/regulatory claim?
  │        │
  ├─► verifier         evidence-gating: does a retrieved passage state each claim?
  │        │
  ▼        ▼
answer + evidence + gaps + flags + verdict + trace
```

**It is sequential, and it says so.** Spec 4.2 says retrieval and expertise-gap
"run concurrently"; spec 4.2's own code sketch feeds retrieval's candidates *into*
expertise-gap, which makes them sequentially dependent. Both cannot be true. The
gap agent needs the candidates (they *are* the targets it scores against) and
deliberately does not re-retrieve, so we run in order and report
`concurrency: "sequential"`. The alternative — a `ThreadPoolExecutor` wrapped
around a real dependency — would put fake parallelism in the trace, which is the
category of thing this spec exists to prevent. Full reasoning, including what
splitting the agent would actually buy (<1 s) and cost, is in
`orchestrator.py`'s docstring.

---

## The agents

| agent | job | abstains when | confidence means |
|---|---|---|---|
| `retrieval` | wraps `07_retrieve_rank.py::Retriever` — no retrieval logic lives here | zero candidates | top candidate's **cross-encoder rerank score**. A relevance score used as a proxy. **Not calibrated.** In `--mock` it is an RRF rank artifact and says so. |
| `expertise_gap` | the differentiator: `assigned_to` portfolio → KMeans footprint → targets that are far from it → who owns them (`invented_by` → `affiliated_with`) | the query names no org (~55%); or the org owns <5 technologies (a footprint built from 3 documents is sparsity, not evidence) | how much **evidence was available**, not how likely the analysis is right. Capped at 0.85 — an unevaluated heuristic cannot earn certainty. |
| `synthesis` | 08's strict-RAG answer step, re-pointed at z.ai/glm-4.6 | no candidates; none carry evidence; top rerank confidence < 0.35 | retrieval's confidence, passed through |
| `policy` | answer-safety (spec 4.5**a**) | the exchange makes no clinical/efficacy/regulatory claim | severity of the worst flag. **Uncalibrated.** |
| `verifier` | evidence-gating — a *different* framing attacking synthesis's answer | nothing was synthesized (no claim to gate) | fraction of claims tied to a retrieved doc. An **uncalibrated proportion**, not P(answer is right). |

**Abstention is a first-class outcome** (`ok=True, abstained=True`). An agent that
always produces output is an agent that hallucinates on the ~55% of queries where
it has no signal. The trace shows declines as declines, with the reason.

---

## What is measured and what is not

**Measured** (numbers live next to the code that relies on them):

- Retrieval, dense-only vs RRF(dense, keyword): recall@100 **0.7736 vs 0.6798** on
  the 123 LLM-adjudicated utility-qrels queries. Keyword is **off** — RRF evicted
  578 relevant docs dense had already found to seat keyword's 24 unique ones.
- Graph as a *retrieval* channel: **zero** unique relevant documents across 62
  firing queries; −0.0243 nDCG. **Off.** This is not an argument against the
  graph — it is an argument against the graph as a retrieval channel. The gap
  agent is the same graph doing the job it is shaped for.
- Graph facts: 1,489,785 nodes / 4,520,029 edges; `organization:bristol myers
  squibb` → 826 assigned technologies (3,155 across 5 canonical variants);
  `inventor:chen lieping` → 52 `invented_by` technologies.
- Gap threshold **scale** (not its correctness): at cosine 0.40 the rule flags
  15.1% of BMS's own portfolio and 60.1% of random docs. The distributions
  overlap. That is why ownership is checked before the centroid test, and why
  every gap carries `nearest_portfolio_doc_cos` as a second opinion.
- Policy false-positive probe: over 20,000 real trial abstracts fed in as if the
  tool had emitted them, dosing fires **4** times (0.02%). This measures only what
  the rules *wrongly* catch. **Recall is unknown.**

**Unmeasured — say so out loud:**

- **The expertise-gap agent has no eval.** There is no judged set of gap analyses,
  so there is no accuracy number and there will not be one until someone builds
  the labels. `payload["unmeasured"]` is `True` on every result it returns. It is
  a **capability demo, not a measured result.** Gaps are leads to check.
- **The verifier's effectiveness is unmeasured.** It reduces the rate at which
  ungrounded claims reach the user. By how much is unknown — there is no labelled
  grounded/ungrounded set here. Do not put a number on it.
- **Every threshold is hand-set**, not tuned: gap cosine 0.40, support 0.65,
  min portfolio 5, max orgs/inventor 10 (the measured p99). No labelled set exists
  to tune against; each one's comment says so.
- **No confidence here is calibrated.** They are never averaged across agents
  (`base.py` invariant 3) — the scales are unrelated, and their mean would be a
  fourth number meaning nothing.

---

## Call it evidence-gating, not verification

The verifier is **an LLM checking an LLM** against a handful of retrieved
passages. It answers *"does this retrieved text state this sentence?"* — never
*"is this sentence true?"*. A claim can be true and land `unsupported` (the corpus
lacks it); a claim can be false and land `supported` (the corpus is wrong). Both
are correct behaviour.

The output key is `verified: bool` because spec 4.2 fixes that name. It means
**the evidence gate passed**. Nothing in this package prints the word "verified"
at a human.

With no `DH_LLM_API_KEY` the verifier falls back to lexical overlap plus a
number/identifier grounding check. That path is **strictly weaker**: it cannot
detect contradiction at all, and cannot credit a faithful paraphrase. It labels
itself `mode: "deterministic"` and ships a `caveat`. It never runs silently.

---

## Two things composition forced, that neither module was wrong about alone

1. **Citations do not appear inline in the answer prose.** The verifier's
   deterministic path flags any digit-bearing token absent from the evidence and
   does not strip URLs first, so `[source: https://patents.google.com/patent/US6803192B1]`
   tokenizes as a fabricated identifier and makes *every* sentence unsupported →
   `fail` → degrade. Copying 08's `_explain_mock` verbatim would have degraded all
   three demo queries, including the two that are supposed to succeed. Citations
   therefore travel **structurally** (one `doc_id` per claim in `payload["claims"]`),
   which still satisfies "every claim carries a citation" and is machine-checkable
   rather than a substring in prose. Pinned by
   `tests/test_synthesis.py::TestTheAnswerSurvivesItsOwnGate`.
   *Corollary:* `payload["answer"]` contains **only** grounded sentences. Every
   caveat and count lives in a sibling field — an aside like "Retrieval returned 5
   documents" introduces the ungrounded token "5" and fails the whole answer.

2. **Gap findings are not merged into the gated prose.** A gap is a claim about
   *absence* ("BMS has no B7-H3 coverage"), derived by graph traversal. No
   retrieved abstract can support it — the document that says so does not exist —
   so folding it into the verifier-checked answer would fail the gate on every run
   and degrade the money shot. The spec's own architecture diagram agrees: its
   output is `answer + evidence + gaps + flags + verdict`, where `gaps` is a
   sibling of `answer`. Gaps travel with their own graph provenance and their own
   unmeasured flag. Synthesis still *receives* them (per spec 4.2's sketch) and
   uses them to foreground the candidates the gap agent flagged.

---

## Known limits worth stating before someone finds them

- **Org identity is fragmented.** `yale university` (340 techs, all trials, **zero
  patents**) and `univ yale` (622, the patents) are separate canonical keys.
  Suffix-stripping merges `bristol myers squibb co` but cannot merge word-order or
  abbreviation variants. A "Yale gaps" query analyses a trials-only footprint.
  Mayo splits the same way. **Fix is entity resolution, not a threshold.**
- **Genentech was unreachable; `entity_link` has since been changed.** The
  reported failure — label "Genentech, Inc.", with a 1-technology node
  (`genentech inc a member of the roche group`) blocking the unique-token alias,
  making a 2,498-technology company invisible — no longer reproduces: on a
  synthetic index rebuilding exactly that collision, `"genentech gaps in oncology"`
  now links `organization:genentech`. `entity_link` gained legal-suffix stripping
  and a Zipf-frequency filter. **Caveat:** that is the *logic* verified on 7 nodes,
  not reachability re-verified against the real 889,047-node surface index, where
  alias uniqueness is a different question. Re-run `demo/validate_portfolios.py`
  before repeating either the old claim or the new one.
- **Expert nodes are near-useless here.** Spec 4.3 step 6's
  `expert --affiliated_with--> org` path **does not exist** — expert nodes have
  zero `affiliated_with` edges. The real path is
  `expert <-investigated_by- tech -assigned_to-> org`, implemented, and it fires
  for only **13.4%** of experts. `inventor_to_orgs` is the load-bearing route.
- **Spec 4.5(b) (IP / freedom-to-operate) is absent, not stubbed.** `docs.jsonl`
  has no `priority_date`, no filing/grant dates, and `extra{}` is empty, so patent
  expiry is not computable. Policy deliberately does **not** fire on IP queries:
  an IP-triggered run would find zero flags and render as "policy: fired, clean",
  which reads as *the IP was checked and it's fine*. It wasn't.
- **`signals_fired` is derived, not observed** — inferred from which score fields
  are non-zero. An arm that ran but scored everything 0.0 reads as "not fired".
- **The demo's refusal query no longer trips the org linker.** It used to resolve
  the token "what" to `organization:what 3 things joint venture llc` (1 technology),
  so expertise-gap fired and then abstained on the portfolio-size floor. After the
  `entity_link` fix it links nothing, so the agent declines at the *router*. Both
  are honest traces; expect the second. The `MIN_PORTFOLIO_FOR_GAP` floor is still
  load-bearing — 53.6% of org nodes own exactly one technology.
- **`payload["linked"]` is `False` on every real-mode query** unless `DH_USE_GRAPH=1`,
  because the graph arm is off. Do not wire a demo assertion to it.
- **`tests/test_pipeline_smoke.py` is red on `main`** and was before this build: it
  shells out to `01_download_data.py`, which is now `01_download_data_v2.py`.
  Unrelated to this package.

---

## Scorecard (spec section 7, updated)

| Claim | Spec's "after this build" | What actually exists now |
|---|---|---|
| Retrieval agent | ✅ shipped, measured | ✅ shipped, measured — wrapped, unchanged |
| Reranking agent | ✅ shipped, measured | ✅ shipped, measured — inside 07, drives synthesis's confidence |
| Expertise-gap agent | ✅ built, graph-backed, **unmeasured** | ✅ built, graph-backed, **unmeasured**. Portfolios hand-validated on 5 orgs; 3 of the 5 have real reachability/fragmentation problems, listed above |
| Policy/safety agent | ⚠️ scope (a) only, or omit | ⚠️ **scope (a) only.** (b) is not buildable — no `priority_date`. Blocks on dosing; everything else warns |
| Multi-agent orchestration | ✅ real: routing, fan-out, gating, tracing | ✅ real: routing, gating, tracing — **fan-out is sequential and reported as such**, because the spec's own sketch makes the two agents dependent |
| Multi-agent verification | ✅ real, but call it evidence-gating | ✅ real, **evidence-gating**, adversarially framed, gate computed in code and never by the model |

Six claims: four solid, one qualified, one honest about its limits — and one
(fan-out) where the spec asked for something its own code sketch made impossible,
so we did the honest version and wrote down why.

---

## Files

| file | what |
|---|---|
| `base.py` | `AgentResult` / `Agent` / `run_agent` / `to_trace_row`. Three invariants: abstention is success; one agent failing never kills the run; confidences are never averaged. |
| `llm.py` | Optional OpenAI-compatible client for glm-4.6 via z.ai. **Optional in the strong sense**: no key → `.available` is False → every agent falls back and says so. No secret is hardcoded or logged. |
| `graph_index.py` | 286 MB CSR view of the merged graph; loads in ~1 s. Build: `python -m agents.graph_index`. |
| `retrieval.py` | 07 wrapper. Lazy — constructing it loads nothing. |
| `expertise_gap.py` | The differentiator. mmaps `doc_vectors.npy`; materializes only portfolio rows. |
| `synthesis.py` | 08's answer step via z.ai. Deterministic fallback = verbatim quotation, not synthesis. |
| `policy.py` | Answer-safety rules. The LLM may widen the review, never touch the gate. |
| `verifier.py` | Evidence-gating. Adversarial framing, deliberately divergent from synthesis's — a verifier sharing the generator's framing shares its blind spots and returns agreement theatre. |
| `orchestrator.py` | Route → (sequential) fan-out → aggregate → gate → trace. |
| `LLM_CONTRACT.md` | The **verified** glm-4.6 call contract. Read before touching `llm.py` — reasoning tokens bill against `max_tokens` and silently return `content: ""`. |

### Memory

A real Retriever load is ~20 GB (faiss 6.17 GB + doc_vectors 6.17 GB + 603k docs +
bm25) on a 30 GB box. **Do not construct two.** The gap agent adds ~1.6 GB peak and
mmaps the vectors. Constructing any agent, or the orchestrator, loads nothing —
the cost is paid on first `run()`. Unit tests build tiny synthetic fixtures and
never touch a real artifact.
