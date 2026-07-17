# Multi-agent pipeline — build & demo spec

**Status: this describes something that does not exist yet.**
Pitch pp. 5, 11, 13 and the Exec Summary describe it in the present tense. This document
is the plan to make that true, and the honest language to use until it is.

---

## 0. Read this before the deck ships

| Pitch claim | Reality today |
|---|---|
| "Multi-agent pipeline (retrieval, **expertise-gap**, reranking, **policy/safety**)" | Retrieval ✅ and reranking ✅ exist. **Expertise-gap: does not exist.** **Policy/safety: does not exist.** |
| "**multi-agent verification**" | **Does not exist.** One RAG call, one confidence gate, no second opinion. |
| "multi-agent orchestration" | **Does not exist.** `08_multiagent_rag.py` is named "multiagent" but is a single-pass RAG. The word is in the filename, not the architecture. |

Actual serving path:

```
query → FAISS dense (+BM25) → BGE cross-encoder rerank → RAG answer + confidence gate
```

That is **two** of four claimed agents, no orchestration, no verification. `08`'s filename
is the only multi-agent thing about it — and a filename is exactly the kind of artifact a
technical diligence read finds in ten minutes.

### If the deck ships before the build

Do not say "multi-agent pipeline" in the present tense. Say what is true and what is next:

> "Today: a two-stage retrieve-and-rerank pipeline with an evidence-gated RAG layer.
> On deck: an agent architecture that adds expertise-gap analysis over our knowledge
> graph, a policy layer, and cross-agent verification."

That is a stronger sentence than the current one, because it survives the follow-up
question. The current one does not.

**The claim is not a lie — it is a roadmap in the present tense.** The fix is tense, or
code. This document is the code path.

---

## 1. What already exists (reuse, do not rebuild)

| Component | Where | Notes |
|---|---|---|
| Retrieval | `prod/07_retrieve_rank.py::Retriever` | dense + BM25 + graph → RRF |
| Reranking | same, `_rerank()` | BGE cross-encoder, 512 ctx |
| RAG + confidence gate | `08_multiagent_rag.py` | single pass; the seam to build on |
| LLM | glm-4.6 via z.ai | `DH_LLM_BASE_URL / API_KEY / MODEL` |
| **Knowledge graph** | `data_merged/graph/` | **1.49M nodes / 4.52M edges** |
| Graph node types | — | `technology`, `inventor`, `organization`, `expert`, `facility` |
| Graph relations | — | `invented_by`, `assigned_to`, `affiliated_with` |
| Graph embeddings | `data_merged/artifacts/rgcn_node_emb.npy` | 128-dim, 1.49M nodes |
| Entity linking | `discovery_hub/entity_link.py` | ~45% query fire rate, abstains cleanly |
| Corpus | `data/normalized/docs.jsonl` | 600,738 docs: patents / trials / papers / SBIR |

### The insight that makes this worth building

The graph does **not** help retrieval. Measured: −0.0243 nDCG in a correctly-configured
system, costing 1.2 s and 0.068 recall. It is a bad retrieval channel.

**But retrieval was the wrong job for it.** "What does this company cover, what does it
*not* cover, and who covers the gap" is a *graph traversal* question, not a nearest-
neighbour question. `org --assigned_to--> technology` and `expert --affiliated_with-->
org` are exactly the edges that answer it.

So the expertise-gap agent is not a new claim bolted on to satisfy a slide. It is the
first job this graph has been given that it is actually shaped for — and it turns the
graph from a measured liability into the thing no keyword search can do.

---

## 2. Architecture

```
                          ┌─────────────────┐
   scout query ─────────► │  Orchestrator   │
                          └────────┬────────┘
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │  Retrieval   │    │ Expertise-   │    │   Policy /   │
      │    agent     │    │  gap agent   │    │ safety agent │
      │  (EXISTS)    │    │   (BUILD)    │    │   (BUILD)    │
      └───────┬──────┘    └───────┬──────┘    └───────┬──────┘
              │                   │                   │
              ▼                   │                   │
      ┌──────────────┐            │                   │
      │  Reranking   │            │                   │
      │    agent     │            │                   │
      │  (EXISTS)    │            │                   │
      └───────┬──────┘            │                   │
              └───────────┬───────┴───────────────────┘
                          ▼
                 ┌──────────────────┐
                 │  Synthesis (RAG) │   ← 08, largely as-is
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │  Verifier agent  │   ← "multi-agent verification"
                 │     (BUILD)      │
                 └────────┬─────────┘
                          ▼
                    answer + evidence + gaps + flags + verdict
```

Four agents run; two of them already exist. The new work is three components and a
router.

---

## 3. Build order

Ordered by (value × defensibility) ÷ effort. **Do not build all of it before the demo.**

| # | Component | LoC | Effort | Demo-critical |
|---|---|---:|---|---|
| 1 | `agents/base.py` — Agent protocol, trace, timing | ~80 | 1 h | yes |
| 2 | `agents/orchestrator.py` — routing, fan-out, aggregation | ~200 | 3 h | yes |
| 3 | `agents/retrieval.py` — wrap `07` | ~60 | 1 h | yes |
| 4 | `agents/expertise_gap.py` — **the differentiator** | ~250 | 6 h | **yes** |
| 5 | `agents/verifier.py` — claim ↔ evidence check | ~150 | 3 h | **yes** |
| 6 | `agents/policy.py` — IP/regulatory/claim flags | ~200 | 4 h | no (v2) |
| 7 | `demo/run_demo.py` — scripted CLI walkthrough | ~120 | 2 h | yes |

**Minimum viable demo = 1, 2, 3, 4, 5, 7 ≈ 16 h of focused work.** That is two days, not
an evening. Plan accordingly, and do not promise a demo you have not rehearsed end to end.

---

## 4. Agent specs

### 4.1 `agents/base.py`

```python
@dataclass
class AgentResult:
    agent: str
    ok: bool
    payload: dict
    evidence: list[dict]        # [{doc_id, quote_span, source_url}]
    confidence: float           # 0-1, calibrated per agent
    latency_ms: float
    abstained: bool = False     # abstention is a FIRST-CLASS outcome, not a failure
    error: str | None = None

class Agent(Protocol):
    name: str
    def run(self, query: str, ctx: dict) -> AgentResult: ...
```

**Every agent must be able to abstain.** The graph already does this correctly
(`entity_link` returns nothing → no signal). An agent that always produces output is an
agent that hallucinates on 55% of queries. Abstention is the feature.

### 4.2 `agents/orchestrator.py`

Responsibilities, in order:

1. **Route.** Not every agent runs on every query. Expertise-gap only fires when the query
   names an organization (`entity_link`, ~45%). Policy only fires when the answer makes a
   regulatory or IP claim.
2. **Fan out.** Retrieval and expertise-gap are independent — run concurrently
   (`ThreadPoolExecutor`, both are I/O-bound).
3. **Aggregate.** Never average confidences across agents. Carry them separately, the same
   discipline as never averaging BGE and Qwen scores.
4. **Gate.** If the verifier fails, the orchestrator degrades the answer to
   "evidence-only" — retrieved documents, no synthesis. **Never ship an unverified claim.**
5. **Trace.** Emit a per-agent record: fired / abstained / latency / confidence. This is
   the demo's most persuasive artifact — it shows the architecture *working*, including
   the parts that correctly decline.

```python
class Orchestrator:
    def run(self, query: str) -> dict:
        trace = []
        r = self.retrieval.run(query, {})                 # always
        gaps = None
        if self.expertise_gap.should_fire(query):         # ~45%
            gaps = self.expertise_gap.run(query, {"candidates": r.payload["candidates"]})
        answer = self.synthesis.run(query, {"candidates": r.payload["candidates"],
                                            "gaps": gaps})
        v = self.verifier.run(query, {"answer": answer, "evidence": r.evidence})
        if not v.ok:
            answer = self._degrade_to_evidence_only(r)    # the gate
        return {"answer": answer, "trace": trace, "verified": v.ok}
```

### 4.3 `agents/expertise_gap.py` — the differentiator

**Question it answers:** *"Company X wants to move into indication Y. What do they already
cover, what are they missing, and which university invention fills it?"*

That is the actual scouting job, and no keyword search does it.

**Algorithm** (all from the merged graph, no new data):

```
1. link the query to an org node          entity_link.link_query()
2. portfolio = {tech : org --assigned_to--> tech}     ← what they own
3. embed each portfolio tech (existing doc vectors), cluster
   → the company's capability footprint
4. target = the query's therapeutic area; retrieve its top-k technologies
5. gap = target technologies whose nearest portfolio cluster is FAR
   → "this company has no coverage here"
6. for each gap, walk the graph for who DOES cover it:
     tech <--assigned_to-- org        (competitor / licensor)
     tech <--invented_by-- inventor --affiliated_with--> org
     expert --affiliated_with--> org  ← the `expert` node type exists; use it
7. LLM (glm-4.6) renders the gap + the fillers as a paragraph, cites doc_ids
```

**Abstains** when the query names no org (~55%). Say so in the output — a scouting tool
that admits "you didn't name a company, so I can't do gap analysis" reads as trustworthy.

**Validation before it is demoed.** Pick 5 orgs whose portfolios you can check by hand
(Mayo, Yale, BMS are in the graph and verifiable on Google Patents). If the portfolio
extraction is wrong for those five, it is wrong everywhere, and a partner who knows pharma
will spot it instantly.

### 4.4 `agents/verifier.py` — "multi-agent verification"

A second LLM pass with a different prompt and a different job: **check the first one.**

```
input : the synthesized answer + the retrieved evidence
task  : for EACH factual claim in the answer —
          - is it supported by a specific retrieved document? cite doc_id
          - is it contradicted by any retrieved document?
          - is it unsupported (present in neither)?
output: {claims: [{text, status: supported|contradicted|unsupported, doc_id}],
         verdict: pass|fail, unsupported_count}
gate  : any unsupported claim → verdict=fail → orchestrator degrades to evidence-only
```

Use structured output. Use a **different prompt framing** from the synthesis agent — a
verifier that shares the generator's framing shares its blind spots, and you get agreement
theatre rather than verification.

**This is the cheapest real agent to build and the easiest to demo.** Ask it a question
where the corpus has no answer, and watch it refuse. That is worth more in the room than
any successful answer.

### 4.5 `agents/policy.py` — v2, not demo-critical

Two candidate scopes. **Pick one; do not hand-wave both.**

**(a) Answer-safety** — no new data needed:
- flags clinical/efficacy claims stated as fact
- flags dosing or treatment guidance (out of scope for a scouting tool)
- flags overstated development stage ("approved" vs "Phase 2")

**(b) IP / freedom-to-operate** — needs patent metadata you may not have:
- patent expiry (priority date + 20 y) — **verify `priority_date` is in `docs.jsonl` first**
- assignee ≠ inventor's current affiliation (licensing complexity)
- family members across jurisdictions

**(a) is honest and buildable today. (b) depends on fields that may not exist** —
`doc_views.py` assumes `claims`, `cpc`, `priority_date` for patents but explicitly notes
they are not guaranteed. Check before designing around them.

---

## 5. The demo

### Setup

```bash
export DH_LLM_BASE_URL='https://api.z.ai/api/paas/v4'
export DH_LLM_MODEL='glm-4.6'
export DH_LLM_API_KEY='...'
export DH_GRAPH_DIR=/home/alex/discovery_hub/data_merged/graph
# ship the measured config, not the shipped one:
export DH_USE_KEYWORD=0      # +0.30 nDCG, 6x clear of any power concern
```

### Script — three queries, in this order

**1. The hit.** Query: `"monoclonal antibody targeting PD-L1 for oncology"`
- Retrieval surfaces `US6803192B1` "B7-H1, a novel immunoregulatory molecule"
- Show the trace: dense fired, graph abstained (no org named), expertise-gap abstained
- **Point at the abstentions.** "Two agents declined. That's the design."
- The answer cites B7-H1 = PD-L1, older name. No keyword search finds this.

**2. The differentiator.** Query: `"Bristol Myers Squibb gaps in B7 family immunotherapy"`
- Expertise-gap **fires** (org linked)
- Portfolio: BMS's B7-family technologies from the graph
- Gap: B7-H3 / B7-H4 / B7-H5 coverage they lack
- Fillers: Chen Lieping's patents, assigned to Mayo/Yale, reachable via `invented_by`
- **This is the money shot.** It is a licensing lead, produced by graph traversal, from
  real data. Show the subgraph (`chen_pdl1_story.svg`) next to it.

**3. The refusal.** Query: `"what dose of pembrolizumab should I give a stage IV NSCLC patient"`
- Retrieval finds documents; synthesis drafts; **verifier fails it**; policy flags it
- Orchestrator degrades to evidence-only
- **A tool that refuses is a tool a pharma partner can put in front of their legal team.**
  This slide sells better than a correct answer.

### What to show on screen

The **trace table**, not the prose. Per agent: fired / abstained / confidence / ms.
The architecture is invisible in an answer and obvious in a trace.

| agent | status | conf | ms |
|---|---|---:|---:|
| retrieval | fired | 0.81 | 842 |
| expertise-gap | fired | 0.64 | 1203 |
| policy | abstained | — | 2 |
| verifier | **pass** | 0.92 | 890 |

---

## 6. What not to claim, even after building this

- **Not "verified" — "evidence-gated."** The verifier is an LLM checking an LLM. It
  reduces unsupported claims; it does not establish truth. Say what it does.
- **No accuracy number without a benchmark.** There is no eval for the expertise-gap
  agent. Building one means judged gap-analyses, which do not exist. Until then it is a
  capability demo, not a measured result.
- **The graph's retrieval contribution is measured and negative.** If the deck now claims
  the graph powers expertise-gap, that is defensible. If it claims the graph improves
  *search*, that contradicts your own data.
- **`08_multiagent_rag.py` is not evidence of a multi-agent system.** Rename it or build
  it. A filename is not an architecture, and diligence reads filenames.

---

## 7. Honest scorecard for the deck, after the build

| Claim | After this build |
|---|---|
| Retrieval agent | ✅ shipped, measured |
| Reranking agent | ✅ shipped, measured |
| Expertise-gap agent | ✅ built, graph-backed, **unmeasured** |
| Policy/safety agent | ⚠️ scope (a) only, or omit |
| Multi-agent orchestration | ✅ real: routing, fan-out, gating, tracing |
| Multi-agent verification | ✅ real, but call it evidence-gating |

Six claims, four solid, one qualified, one honest about its limits. That is a better slide
than four overclaims — and it is the version that survives the second question.
