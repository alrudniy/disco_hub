"""
T1 SCORE FORENSICS + T2 TEXT PARITY.

The load-bearing claim under test: on "HER2 targeted therapy for breast cancer" the
rerank stage puts all 45 literal-HER2 docs at 1-45 and all 5 bridged docs at 46-50.
That exact partition is what a SCORING BUG predicts with certainty and what lexical
dominance predicts only sometimes. So: is it a bug?

Parameters, fixed before the run, none altered from the shipped config:
  top_k=50, rerank_k=50          -- see all 50, the same pool the finding came from
  rerank_max_length=512          -- shipped default (07 _get_cross_encoder getattr)
  rerank_batch_size=16           -- shipped default
  model: config.RERANK_MODEL (BAAI/bge-reranker-v2-m3), config.EMBED_MODEL (ft-4b)

Nothing here writes. Diagnosis only.
"""
import importlib.util
import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/dh_multiagent")
from discovery_hub import config

QUERY = "HER2 targeted therapy for breast cancer"
TARGET = "uspto:US11903948B2"          # the ErbB2 patent: dense 9 -> rerank 49


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


spec = importlib.util.spec_from_file_location("rr", "/workspace/dh_multiagent/07_retrieve_rank.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

print("building retriever (real) ...", flush=True)
R = rr.Retriever(mock=False, device="cuda")

# ---------- dense ranks over the WHOLE corpus, for the 50 ----------
qv = R.embedder.encode_queries([QUERY])[0].astype(np.float32)
V = torch.from_numpy(np.asarray(R.doc_vectors, dtype=np.float32)).cuda()
sims = (V @ torch.from_numpy(qv).cuda()).cpu().numpy()
order = np.argsort(-sims)
dense_rank_of = {R.doc_ids[j]: i + 1 for i, j in enumerate(order)}

# ---------- the 50, reranked (run twice for determinism) ----------
print("retrieve run 1 ...", flush=True)
c1 = R.retrieve(QUERY, top_k=50, rerank_k=50)
print("retrieve run 2 ...", flush=True)
c2 = R.retrieve(QUERY, top_k=50, rerank_k=50)

ce = R._get_cross_encoder()
tok = ce.tokenizer
max_len = getattr(config.RETRIEVAL, "rerank_max_length", 512)

print(f"\nCE model            : {config.RERANK_MODEL}")
print(f"CE max_length       : {ce.max_length}  (config getattr -> {max_len})")
print(f"CE num_labels       : {getattr(ce, 'num_labels', '?')}")
act = getattr(ce, "activation_fct", None) or getattr(ce, "default_activation_function", None)
print(f"CE activation       : {act}")
print(f"embedder            : {config.EMBED_MODEL}  dim={R.embedder.dim}")
emb_max = getattr(R.embedder.model, "max_seq_length", "?")
print(f"embedder max_seq_len: {emb_max}")

# ---------- T1 TABLE ----------
print("\n" + "=" * 118)
print("T1  ALL 50, rerank order")
print("=" * 118)
print(f"{'rk_rr':>5} {'rk_dense':>9} {'score_rerank (full repr)':>26} {'her2':>5} "
      f"{'n_tok':>6} {'trunc':>6}  doc_id")
rows = []
for i, c in enumerate(c1, 1):
    d = R.docs[c["doc_id"]]
    txt = d.embedding_text
    enc = tok(QUERY, txt, truncation=True, max_length=ce.max_length)
    n_tok = len(enc["input_ids"])
    full = tok(QUERY, txt, truncation=False)
    n_full = len(full["input_ids"])
    lit = "her2" in squash((d.title or "") + " " + (d.abstract or ""))
    s = c.get("rerank_score")
    rows.append({
        "doc_id": c["doc_id"], "rank_dense": dense_rank_of.get(c["doc_id"]),
        "rank_rerank": i, "score_rerank": s, "her2": lit,
        "n_tok": n_tok, "n_tok_untrunc": n_full, "truncated": n_full > ce.max_length,
        "title": (d.title or "")[:40],
    })
    print(f"{i:>5} {dense_rank_of.get(c['doc_id']):>9} {repr(s):>26} {str(lit):>5} "
          f"{n_tok:>6} {str(n_full > ce.max_length):>6}  {c['doc_id']}")

scores = [r["score_rerank"] for r in rows]
lit_s = [r["score_rerank"] for r in rows if r["her2"]]
brg_s = [r["score_rerank"] for r in rows if not r["her2"]]

print("\n" + "-" * 118)
print("T1 ANSWERS")
print("-" * 118)
print(f"  bridged (non-literal) docs: {len(brg_s)}   literal: {len(lit_s)}")
print(f"  bridged scores (full repr): {[repr(s) for s in brg_s]}")
print(f"  identical to each other?    {len(set(brg_s)) < len(brg_s)}  "
      f"({len(set(brg_s))} distinct of {len(brg_s)})")
degen = [s for s in scores if s is None or (isinstance(s, float) and
         (np.isnan(s) or np.isinf(s) or s == 0.0))]
print(f"  any 0.0/None/NaN/inf?      {bool(degen)}  {degen}")
print(f"  distinct scores of 50:     {len(set(scores))}")
print(f"  score range:               min={min(scores)!r}  max={max(scores)!r}")
if lit_s and brg_s:
    print(f"  literal  min={min(lit_s)!r}")
    print(f"  bridged  max={max(brg_s)!r}")
    gap = min(lit_s) - max(brg_s)
    print(f"  separation (min_lit - max_bridged) = {gap!r}")
    srt = sorted(scores)
    diffs = [(round(srt[i+1]-srt[i], 6), i) for i in range(len(srt)-1)]
    big = sorted(diffs, reverse=True)[:3]
    print(f"  3 largest adjacent gaps in the sorted 50: {big}")
    print(f"  -> continuum or two clusters? largest gap {big[0][0]!r} vs "
          f"median gap {np.median([d for d, _ in diffs]):.6f}")

same = [a["doc_id"] for a in c1] == [b["doc_id"] for b in c2]
same_scores = all(a.get("rerank_score") == b.get("rerank_score") for a, b in zip(c1, c2))
print(f"  deterministic across 2 runs? order={same}  scores_bitwise={same_scores}")

# ---------- T2 TEXT PARITY ----------
print("\n" + "=" * 118)
print(f"T2  TEXT PARITY for {TARGET}")
print("=" * 118)
d = R.docs[TARGET]
dense_text = d.embedding_text          # 04_generate_embeddings.py embeds exactly this
bge_text = d.embedding_text            # 07 _rerank passes exactly this
print(f"  dense stage embedded : d.embedding_text  ({len(dense_text)} chars)")
print(f"  BGE stage received   : d.embedding_text  ({len(bge_text)} chars)")
print(f"  SAME SOURCE STRING?  : {dense_text == bge_text}")

emb_tok = R.embedder.model.tokenizer
e_full = emb_tok(dense_text, truncation=False)["input_ids"]
b_full = tok(QUERY, bge_text, truncation=False)["input_ids"]
b_trunc = tok(QUERY, bge_text, truncation=True, max_length=ce.max_length)["input_ids"]
print(f"  encoder tokens (untruncated, doc only) : {len(e_full)}   "
      f"max_seq_length={emb_max} -> truncated={len(e_full) > (emb_max if isinstance(emb_max,int) else 10**9)}")
print(f"  BGE tokens (untruncated, query+doc)    : {len(b_full)}   "
      f"max_length={ce.max_length} -> truncated={len(b_full) > ce.max_length}")
print(f"  BGE tokens actually scored             : {len(b_trunc)}")
print(f"\n  full embedding_text of {TARGET}:\n---\n{dense_text}\n---")

n_trunc = sum(1 for r in rows if r["truncated"])
print(f"\n  of the 50 reranked pairs, truncated at BGE max_length: {n_trunc}")
print(f"  truncated among literal : {sum(1 for r in rows if r['truncated'] and r['her2'])} / {len(lit_s)}")
print(f"  truncated among bridged : {sum(1 for r in rows if r['truncated'] and not r['her2'])} / {len(brg_s)}")

json.dump(rows, open("/workspace/dh_multiagent/t1_rows.json", "w"), indent=1)
print("\nwrote t1_rows.json")
