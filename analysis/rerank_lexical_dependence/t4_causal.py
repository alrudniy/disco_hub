"""
T4 CAUSAL PROBE. Single-term intervention, every other byte constant.

Runs (b) FIRST, per instruction.
  (b) uspto:US12227591B2 -- rerank rank 1, n_tok=319 (>= 50-doc median 212.5, so no
      sparse-evidence confound), z=+0.530. Replace HER2 -> ErbB2, all case/separator
      variants. Rescore, same query.
  (a) uspto:US11903948B2 -- the ErbB2 patent, dense 9 -> rerank 49, z=-6.480.
      Replace ErbB2 -> HER2, all case/separator variants. Rescore, same query.

Reported as delta-LOGIT (z = ln(s/(1-s))) and delta-RANK. Not delta-sigmoid.

PRE-REGISTERED CRITERION (stated before this run, from the task):
  observed boundary = 0.90 logits, a LOWER BOUND on any constant lexical bonus b.
    both swaps >= 0.90, similar magnitude -> constant lexical bonus, sufficient
    (a) < 0.90                           -> bonus cannot produce the partition
    (a) and (b) disagree in magnitude    -> context interaction, different claim
Report the numbers, not the verdict.

Parameters: unchanged shipped config. BGE bge-reranker-v2-m3, max_length=512,
batch_size=16, Sigmoid head, query "HER2 targeted therapy for breast cancer".
Nothing is written. Diagnosis only.
"""
import importlib.util
import json
import math
import re
import sys

sys.path.insert(0, "/workspace/dh_multiagent")
from discovery_hub import config

QUERY = "HER2 targeted therapy for breast cancer"
DOC_B = "uspto:US12227591B2"      # rank 1, token-rich  -> HER2 becomes ErbB2
DOC_A = "uspto:US11903948B2"      # rank 49, the bridge -> ErbB2 becomes HER2

# All case + separator variants. \b anchors so "other 2" / "superb 2" cannot match.
RE_HER2 = re.compile(r"\bher[\s\-‐‑‒–—_]*2\b", re.IGNORECASE)
RE_ERBB2 = re.compile(r"\berb[\s\-_]?b[\s\-‐‑‒–—_]*2\b", re.IGNORECASE)

logit = lambda s: math.log(s / (1.0 - s))

spec = importlib.util.spec_from_file_location("rr", "/workspace/dh_multiagent/07_retrieve_rank.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

print("building retriever ...", flush=True)
R = rr.Retriever(mock=False, device="cuda")
ce = R._get_cross_encoder()
tok = ce.tokenizer

rows = json.load(open("/workspace/dh_multiagent/t1_rows.json"))
base_z = {r["doc_id"]: logit(r["score_rerank"]) for r in rows}
base_s = {r["doc_id"]: r["score_rerank"] for r in rows}


def rank_of(doc_id, new_score):
    """Rank the doc among the 50 with only its own score changed."""
    scores = {r["doc_id"]: r["score_rerank"] for r in rows}
    scores[doc_id] = new_score
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return 1 + [d for d, _ in order].index(doc_id)


def probe(tag, doc_id, pattern, replacement):
    d = R.docs[doc_id]
    orig = d.embedding_text
    new, n = pattern.subn(replacement, orig)
    s_orig = float(ce.predict([(QUERY, orig)], batch_size=16)[0])
    s_new = float(ce.predict([(QUERY, new)], batch_size=16)[0])
    z_o, z_n = logit(s_orig), logit(s_new)
    r_o, r_n = rank_of(doc_id, s_orig), rank_of(doc_id, s_new)
    print("\n" + "=" * 90)
    print(f"T4({tag})  {doc_id}   substitutions made: {n}")
    print("=" * 90)
    print(f"  title (orig): {(d.title or '')[:78]}")
    print(f"  title (new) : {pattern.sub(replacement, d.title or '')[:78]}")
    print(f"  n_tok orig/new: {len(tok(QUERY, orig)['input_ids'])} / {len(tok(QUERY, new)['input_ids'])}")
    print(f"  chars orig/new: {len(orig)} / {len(new)}")
    print(f"  score  {s_orig!r}  ->  {s_new!r}")
    print(f"  logit  {z_o:+.4f}  ->  {z_n:+.4f}     DELTA-LOGIT = {z_n - z_o:+.4f}")
    print(f"  rank   {r_o}  ->  {r_n}               DELTA-RANK  = {r_n - r_o:+d}")
    # sanity: verify the swap really removed/added the literal token
    sq = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    print(f"  squashed contains 'her2'  orig={('her2' in sq(orig))}  new={('her2' in sq(new))}")
    print(f"  squashed contains 'erbb2' orig={('erbb2' in sq(orig))} new={('erbb2' in sq(new))}")
    return z_n - z_o, r_n - r_o, n


# (b) FIRST, per instruction.
db, drb, nb = probe("b", DOC_B, RE_HER2, "ErbB2")
da, dra, na = probe("a", DOC_A, RE_ERBB2, "HER2")

print("\n" + "=" * 90)
print("T4 SUMMARY  (criterion pre-registered; boundary = 0.8951 logits)")
print("=" * 90)
print(f"  (b) rank-1 rich doc,  HER2 -> ErbB2 : delta-logit {db:+.4f}   delta-rank {drb:+d}   ({nb} subs)")
print(f"  (a) ErbB2 bridge doc, ErbB2 -> HER2 : delta-logit {da:+.4f}   delta-rank {dra:+d}   ({na} subs)")
print(f"\n  |delta(a)| = {abs(da):.4f}   vs boundary 0.8951  -> "
      f"{'>= boundary' if abs(da) >= 0.8951 else '< boundary'}")
print(f"  |delta(b)| = {abs(db):.4f}   vs boundary 0.8951  -> "
      f"{'>= boundary' if abs(db) >= 0.8951 else '< boundary'}")
print(f"  magnitude ratio |a|/|b| = {abs(da)/abs(db) if db else float('nan'):.2f}")
print(f"  signs: (a) {'+' if da>0 else '-'}  (b) {'+' if db>0 else '-'}   "
      f"(lexical prediction: a positive, b negative)")
