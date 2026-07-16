#!/usr/bin/env python3
"""
demo/run_demo.py -- the scripted walkthrough from spec section 5.

    source demo/env.sh
    venv/bin/python demo/run_demo.py              # the three scripted queries
    venv/bin/python demo/run_demo.py --mock       # plumbing only, no real index
    venv/bin/python demo/run_demo.py --query '...' [--json] [--k 10]

WHAT TO LOOK AT: the TRACE TABLE, not the prose. The architecture is invisible in
an answer and obvious in a trace -- including, and especially, the agents that
correctly decline. Spec section 5 is explicit about this and it is right.

MEMORY: a real (non-mock) run loads faiss + doc_vectors + bm25 + the corpus
(~20 GB) plus the graph index (~1.6 GB) on a 30 GB box. Run ONE of these at a
time, and do not run it alongside anything else that loads the index.

NOTHING HERE HARDCODES AN EXPECTED RESULT. The three queries carry a note saying
what the spec predicts, and the demo prints what ACTUALLY surfaced. If the money
shot does not fire, the demo shows that it did not fire. A demo that asserts its
own expected output is a screenshot, not a system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm import LLMClient          # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from discovery_hub import config          # noqa: E402
from discovery_hub.determinism import set_global_determinism  # noqa: E402

# The three queries, in spec order. `note` is what the spec PREDICTS -- printed as
# a prediction, next to whatever actually happened, never asserted as the outcome.
SCRIPT = [
    {
        "title": "1. THE HIT THAT ISN'T (the reranker eats the bridge)",
        "query": "HER2 targeted therapy for breast cancer",
        "note": ("The spec's version of this slide claims retrieval surfaces a "
                 "document no keyword search could find. It does not, and the "
                 "reason is the most useful thing in this demo. "
                 "THE DENSE ENCODER GENUINELY BRIDGES NOMENCLATURE: 253 documents "
                 "here say 'ErbB2' and never 'HER2' (any spelling), and dense cosine "
                 "puts uspto:US11903948B2 'Anti-ErbB2 antibody-drug conjugate' at "
                 "rank 9 of 603,369 -- ErbB2 shares no characters with HER2, so that "
                 "is real synonym knowledge, and a HER2 keyword search returns none "
                 "of those 253. THEN THE CROSS-ENCODER UNDOES IT: of the 50 "
                 "candidates reranked for this query, the 45 that literally contain "
                 "'HER2' take positions 1-45 and the 5 that do not take positions "
                 "46,47,48,49,50 -- exactly the last five. That ErbB2 patent lands "
                 "49th. The reranked top-10 is 100% literal-HER2. So the pipeline's "
                 "FINAL stage, whose score this system reports as confidence, ranks "
                 "like the keyword search the pitch says it beats. "
                 "Watch the ABSTENTIONS too -- no org is named, nothing clinical is "
                 "claimed, so two agents decline. That part is the design working. "
                 "(The spec's original query, 'monoclonal antibody targeting PD-L1' "
                 "expecting US6803192B1 'B7-H1', fails harder: rank 7,585, cosine "
                 "0.234 -- no bridge at all. Chosen by sweeping 28 synonym pairs "
                 "under a separator-blind control; see README section 4.)"),
    },
    {
        "title": "2. THE DIFFERENTIATOR",
        "query": "Bristol Myers Squibb gaps in B7 family immunotherapy",
        "note": ("Expertise-gap should FIRE: the org links, the portfolio comes from "
                 "assigned_to edges, and fillers are reachable via invented_by (Chen "
                 "Lieping's patents, assigned to Mayo/Yale). This is graph traversal "
                 "answering a question about ABSENCE -- no keyword search does it. "
                 "It is also UNMEASURED: there is no judged gap-analysis set to score "
                 "it against, so it is a capability demo, not a result."),
    },
    {
        "title": "3. THE REFUSAL",
        "query": "what dose of pembrolizumab should I give a stage IV NSCLC patient",
        "note": ("Retrieval finds documents and synthesis drafts, but policy should "
                 "flag dosing as a BLOCK and the orchestrator should degrade to "
                 "evidence-only. A tool that refuses is a tool a pharma partner can "
                 "put in front of their legal team."),
    },
]

_RULE = "-" * 78


def banner(mock: bool, k: int | None) -> None:
    """The resolved config. Nobody in the room should have to guess what ran."""
    llm = LLMClient()
    print("=" * 78)
    print("  DISCOVERY HUB -- multi-agent demo".center(78))
    print("=" * 78)
    rows = [
        ("mode", "MOCK (plumbing only -- no real index, no real scores)" if mock
                 else "REAL"),
        ("embed model", config.EMBED_MODEL),
        ("embed dim", str(config.EMBED_DIM)),
        ("data root", str(config.DATA_ROOT)),
        ("graph dir", str(config.GRAPH_DIR)),
        ("artifact dir", str(config.ARTIFACT_DIR)),
        ("index dir", str(config.INDEX_DIR)),
        ("keyword arm", "ON" if config.RETRIEVAL.use_keyword else
                        "OFF (measured: dense-only +0.0937 recall@100 vs RRF)"),
        ("graph arm", "ON" if config.RETRIEVAL.use_graph else
                      "OFF (measured: 0 unique relevant docs; -0.0243 nDCG)"),
        ("top_k_rerank", str(k or config.RETRIEVAL.top_k_rerank)),
        ("LLM", f"{llm.model} @ {llm.base_url}" if llm.available else "NOT AVAILABLE"),
        ("seed", str(config.SEED)),
    ]
    for name, value in rows:
        print(f"  {name:<14} {value}")

    if not llm.available:
        # Loud, unmissable, and before any output that might be mistaken for prose
        # a model wrote. Honesty in the room beats a pretty demo.
        print()
        print("  " + "!" * 74)
        print("  !! DH_LLM_API_KEY IS NOT SET -- THE LLM PATH IS NOT ACTIVE.")
        print("  !! synthesis : falls back to VERBATIM QUOTATION of retrieved evidence.")
        print("  !!             Nothing below is written, summarized or paraphrased")
        print("  !!             by a model. It is the corpus, quoted.")
        print("  !! verifier  : falls back to a lexical grounding check. It is STRICTLY")
        print("  !!             WEAKER than the LLM check and cannot detect")
        print("  !!             contradiction at all.")
        print("  !! Both label themselves in their own output. Nothing here pretends")
        print("  !! a model ran.")
        print("  " + "!" * 74)

    if config.EMBED_DIM != 2560 and not mock:
        print()
        print("  ** WARNING: embed dim is not 2560 but doc_vectors.npy/faiss.index are.")
        print("  ** A real run will fail on a dimension mismatch. Did you source demo/env.sh?")
    print()


def trace_table(trace: list[dict]) -> None:
    """
    Spec section 5's table, exactly: | agent | status | conf | ms |.

    Status vocabulary is wider than the spec's two words because the system has
    more than two honest outcomes: `blocked` (policy refused on purpose),
    `fail`/`pass` (the evidence gate's verdict), `not wired`, and `error` (an agent
    actually broke). Collapsing a deliberate refusal into "error" -- which is what
    reading ok=False alone would do -- would make the refusal slide argue against
    itself.
    """
    print(f"  {'agent':<14} {'status':<11} {'conf':>6} {'ms':>9}")
    print(f"  {'-' * 14} {'-' * 11} {'-' * 6} {'-' * 9}")
    for row in trace:
        conf = "--" if row["conf"] is None else f"{row['conf']:.2f}"
        print(f"  {row['agent']:<14} {row['status']:<11} {conf:>6} {row['ms']:>9.1f}")
    print()
    for row in trace:
        if row.get("reason"):
            print(f"  {row['agent']}: {row['reason']}")
        if row.get("error"):
            print(f"  {row['agent']} ERROR: {row['error']}")


def render(out: dict) -> None:
    """Print what actually happened. No expectations, no hardcoded results."""
    print(_RULE)
    trace_table(out["trace"])
    print(_RULE)

    answer = out["answer"]
    if out["degraded"]:
        print("  ANSWER: WITHHELD -- degraded to evidence-only.")
        print(f"  WHY   : {answer['reason']}")
        print("  The synthesis was dropped entirely rather than shown with a caveat:")
        print("  a drafted answer next to a warning is still the sentence people quote.")
        print("  What survives is what retrieval actually returned:")
        for d in answer["documents"][:5]:
            print(f"    - {d['doc_id']}  {d['source_url']}")
            print(f"      \"{d['quote_span'][:150]}\"")
    else:
        mode = answer["mode"]
        label = ("LLM (glm-4.6)" if mode == "llm"
                 else "DETERMINISTIC -- verbatim quotation, NOT model-written")
        print(f"  ANSWER [{label}]:")
        for claim in answer["claims"]:
            print(f"    * {claim['text']}")
            print(f"      -> cites {claim['doc_id']}")
        if answer.get("caveat"):
            print(f"  CAVEAT: {answer['caveat']}")

    print()
    print(f"  evidence-gated: {'PASS' if out['verified'] else 'FAIL / not run'}"
          f"   (verdict={out['verdict']})")
    print("  NB 'evidence-gated', not 'verified': this is an LLM (or a lexical proxy)")
    print("  checking whether a retrieved passage states each claim. It reduces")
    print("  unsupported claims. It does not establish truth.")

    if out["flags"]:
        print()
        print("  POLICY FLAGS:")
        for f in out["flags"]:
            print(f"    [{f.get('severity', '?')}] {f.get('kind')}: "
                  f"{f.get('rationale', f.get('span', ''))}")

    gaps = out["gaps"]
    if gaps:
        print()
        print(f"  EXPERTISE-GAP  (org: {', '.join(gaps.get('org_labels') or [])})")
        print(f"    portfolio: {gaps.get('portfolio_size')} technologies, "
              f"{gaps.get('n_clusters')} capability clusters")
        if gaps.get("canonical_variants_merged"):
            print(f"    canonicalized variants merged: "
                  f"{', '.join(gaps['canonical_variants_merged'])}")
        print(f"    targets: {gaps.get('n_targets_analysed')} analysed of "
              f"{gaps.get('n_targets')} "
              f"({gaps.get('targets_already_owned')} already owned -> excluded)")
        for g in (gaps.get("gaps") or [])[:3]:
            print(f"    GAP {g['doc_id']}  distance={g['distance']} "
                  f"(nearest owned doc cos={g['nearest_portfolio_doc_cos']})")
            print(f"        {g['title'][:100]}")
            fillers = g.get("fillers", {})
            for org in (fillers.get("orgs") or [])[:3]:
                print(f"        filler org      : {org}")
            for inv in (fillers.get("inventors") or [])[:3]:
                print(f"        filler inventor : {inv}")
        print(f"    narration [{gaps.get('narration_mode')}]: "
              f"{(gaps.get('narration') or '')[:400]}")
        print("    UNMEASURED: no judged gap-analysis set exists. These are leads to")
        print("    check, not findings. See gaps.method_note.")
    elif out.get("gaps_abstained_reason"):
        print()
        print(f"  EXPERTISE-GAP ABSTAINED: {out['gaps_abstained_reason']}")
        print("  ^ Point at this. An agent that declines on 55% of queries is an agent")
        print("    that is not hallucinating on 55% of queries.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", help="run one ad-hoc query instead of the script")
    ap.add_argument("--mock", action="store_true",
                    help="plumbing only: no real index, no real scores")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--k", type=int, default=None, help="top_k_rerank override")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    k = args.k or config.RETRIEVAL.top_k_rerank

    if not args.json:
        banner(args.mock, k)

    orch = Orchestrator.from_config(mock=args.mock, k=k)

    if args.query:
        queries = [{"title": "AD-HOC", "query": args.query, "note": ""}]
    else:
        queries = SCRIPT

    results = []
    for item in queries:
        out = orch.run(item["query"])
        results.append(out)
        if args.json:
            continue
        print("=" * 78)
        print(f"  {item['title']}: {item['query']!r}")
        if item["note"]:
            print()
            print(f"  PREDICTED (spec section 5): {item['note']}")
            print("  ACTUAL:")
        print("=" * 78)
        render(out)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    elif not args.query:
        print("=" * 78)
        print("  The trace is the artifact. Four agents, and the ones that declined")
        print("  declined for a reason they can state. That is the architecture.")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
