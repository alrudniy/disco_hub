"""
V1b + V1c. Is query 2's verifier flip a real verdict (H1), verifier non-determinism
(H2), or an empty/unparseable response defaulting to 0.00 (H3, a BUG)?

H3 is the one that matters: this codebase already shipped its mirror image once
(reasoning tokens ate max_tokens, content came back "", empty was treated as an
answer). If empty is treated as verified-false, the withheld prose is a FABRICATED
ABSTENTION on the one query whose story is "declines instead of confabulating".

V1b: 5 orchestrator runs of query 2, LLM live. Per run capture the verifier's RAW
     LLM response (full, unmodified, "" if empty), numerator, denominator, score,
     draft sha1 + char count, verdict, mode.
V1c: take ONE captured draft, re-verify that exact string 5x. Fixed input isolates
     verifier variance (H2) from draft variance (H1).

Instrumentation only -- no thresholds touched, no config changed. LLMClient.chat is
wrapped to record every raw return; verifier calls are tagged by their system prompt.
"""
import hashlib
import importlib.util
import json
import sys

sys.path.insert(0, "/workspace/dh_multiagent")

from agents import llm as llm_mod
from agents import verifier as ver_mod
from agents.expertise_gap import ExpertiseGapAgent
from agents.graph_index import GraphIndex
from agents.llm import LLMClient
from agents.orchestrator import Orchestrator
from agents.policy import PolicyAgent
from agents.retrieval import RetrievalAgent
from agents.synthesis import SynthesisAgent
from agents.verifier import VerifierAgent
from discovery_hub import config

QUERY = "Bristol Myers Squibb gaps in B7 family immunotherapy"
sha1 = lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

# ---- capture every raw LLM return, tag the verifier's by system prompt ----
RAW = []
_orig_chat = LLMClient.chat


def chat_spy(self, messages, **kw):
    out = _orig_chat(self, messages, **kw)
    sysmsg = next((m["content"] for m in messages if m["role"] == "system"), "")
    RAW.append({"is_verifier": sysmsg.startswith("You are a falsification engine"),
                "raw": out, "n_msgs": len(messages)})
    return out


LLMClient.chat = chat_spy

# ---- capture what the verifier was handed and what it returned ----
CALLS = []
_orig_run = VerifierAgent.run


def run_spy(self, query, ctx):
    before = len(RAW)
    res = _orig_run(self, query, ctx)
    raws = [r for r in RAW[before:] if r["is_verifier"]]
    CALLS.append({"answer": ver_mod._answer_text(ctx.get("answer")),
                  "evidence": ctx.get("evidence"), "result": res,
                  "raw": raws[0]["raw"] if raws else "<<no verifier LLM call made>>"})
    return res


VerifierAgent.run = run_spy

print("building agents (real retriever, LLM live) ...", flush=True)
rr_spec = importlib.util.spec_from_file_location(
    "rr", "/workspace/dh_multiagent/07_retrieve_rank.py")
rr = importlib.util.module_from_spec(rr_spec)
rr_spec.loader.exec_module(rr)
R = rr.Retriever(mock=False, device="cuda")

gi = GraphIndex.load(config.ARTIFACT_DIR / "graph_index.npz")
import numpy as np
dv = np.load(config.EMB_DIR / "doc_vectors.npy", mmap_mode="r")
dids = json.loads((config.EMB_DIR / "doc_ids.json").read_text())

orch = Orchestrator(
    retrieval=RetrievalAgent(retriever=R, mock=False),
    expertise_gap=ExpertiseGapAgent(graph=gi, doc_vectors=dv, doc_ids=dids,
                                    llm=LLMClient()),
    synthesis=SynthesisAgent(llm=LLMClient()),
    policy=PolicyAgent(llm=LLMClient()),
    verifier=VerifierAgent(llm=LLMClient()),
)


def summarise(call, tag):
    res = call["result"]
    p = res.payload
    claims = p.get("claims", [])
    sup = [c for c in claims if c.get("status") == "supported"]
    raw = call["raw"]
    return {
        "tag": tag,
        "draft_sha1": sha1(call["answer"]), "draft_chars": len(call["answer"]),
        "numerator": len(sup), "denominator": len(claims),
        "score": res.confidence, "verdict": p.get("verdict"),
        "mode": p.get("mode"), "abstained": res.abstained,
        "caveat": p.get("caveat", ""),
        "raw_is_empty": (raw is None) or (isinstance(raw, str) and not raw.strip()),
        "raw_len": 0 if raw is None else len(raw),
        "raw": raw,
    }


print("\n" + "=" * 100)
print("V1b  5 ORCHESTRATOR RUNS, QUERY 2, LLM LIVE")
print("=" * 100)
v1b = []
for i in range(1, 6):
    CALLS.clear()
    RAW.clear()
    orch.run(QUERY)
    if not CALLS:
        print(f"run {i}: verifier never ran"); continue
    s = summarise(CALLS[-1], f"v1b-run{i}")
    v1b.append(s)
    print(f"\n--- run {i} ---")
    print(f"  draft sha1={s['draft_sha1']}  chars={s['draft_chars']}")
    print(f"  numerator/denominator = {s['numerator']}/{s['denominator']}"
          f"   score={s['score']!r}   verdict={s['verdict']}   mode={s['mode']}")
    print(f"  raw empty? {s['raw_is_empty']}   raw_len={s['raw_len']}")
    if s["caveat"]:
        print(f"  caveat: {s['caveat'][:120]}")
    print(f"  RAW VERIFIER RESPONSE:\n{s['raw']!r}")

print("\n" + "=" * 100)
print("V1b TABLE")
print("=" * 100)
print(f"{'run':>4} {'draft_sha1':>12} {'chars':>6} {'num':>4} {'den':>4} {'score':>7} "
      f"{'verdict':>8} {'mode':>14} {'raw_empty':>10} {'raw_len':>8}")
for s in v1b:
    print(f"{s['tag'][-4:]:>4} {s['draft_sha1']:>12} {s['draft_chars']:>6} "
          f"{s['numerator']:>4} {s['denominator']:>4} {s['score']:>7.2f} "
          f"{str(s['verdict']):>8} {str(s['mode']):>14} {str(s['raw_is_empty']):>10} "
          f"{s['raw_len']:>8}")

drafts = {s["draft_sha1"] for s in v1b}
print(f"\n  distinct draft sha1 across 5 runs : {len(drafts)}  {sorted(drafts)}")
print(f"  distinct verdicts                 : {sorted({str(s['verdict']) for s in v1b})}")
print(f"  denominators                      : {[s['denominator'] for s in v1b]}")
print(f"  any raw empty/unparseable?        : {any(s['raw_is_empty'] for s in v1b)}")
print(f"  any mode == deterministic?        : {[s['mode'] for s in v1b]}")

# ---------------- V1c: fixed input, 5 re-verifications ----------------
print("\n" + "=" * 100)
print("V1c  THE CONTROL -- ONE captured draft, re-verified 5x on FIXED input")
print("=" * 100)
src = next((c for c in [CALLS[-1]] if c), None)
fixed = v1b[-1] if v1b else None
# reuse the last run's exact draft + evidence
call = CALLS[-1]
draft, evid = call["answer"], call["evidence"]
print(f"  fixed draft sha1={sha1(draft)}  chars={len(draft)}")
ver = VerifierAgent(llm=LLMClient())
v1c = []
for i in range(1, 6):
    RAW.clear(); CALLS.clear()
    res = ver.run(QUERY, {"answer": draft, "evidence": evid})
    s = summarise(CALLS[-1], f"v1c-run{i}")
    v1c.append(s)
    print(f"\n--- v1c run {i} ---")
    print(f"  num/den = {s['numerator']}/{s['denominator']}  score={s['score']!r}  "
          f"verdict={s['verdict']}  mode={s['mode']}  raw_empty={s['raw_is_empty']}")
    print(f"  RAW: {s['raw']!r}")

print("\n" + "=" * 100)
print("V1c ANSWERS")
print("=" * 100)
print(f"  fixed draft sha1        : {sha1(draft)}  (identical input all 5 runs)")
print(f"  scores                  : {[s['score'] for s in v1c]}")
print(f"  verdicts                : {[s['verdict'] for s in v1c]}")
print(f"  denominators            : {[s['denominator'] for s in v1c]}")
print(f"  any raw empty?          : {any(s['raw_is_empty'] for s in v1c)}")
print(f"  scores vary on FIXED input? {len(set(s['score'] for s in v1c)) > 1}"
      f"   -> {'H2 (variance in the verifier)' if len(set(s['score'] for s in v1c))>1 else 'stable'}")
print(f"  drafts varied in V1b?       {len(drafts) > 1}"
      f"   -> {'H1 (variance upstream in the draft)' if len(drafts)>1 else 'drafts stable'}")

json.dump({"v1b": [{k: v for k, v in s.items()} for s in v1b],
           "v1c": [{k: v for k, v in s.items()} for s in v1c]},
          open("/workspace/dh_multiagent/v1_results.json", "w"), indent=1, default=str)
print("\nwrote v1_results.json")
