"""
Diagnostic: WHY does the gap agent report zero gaps for BMS in the B7 family?

Two hypotheses, and they demand opposite responses:
  (H1) BMS genuinely covers B7. The spec's premise is wrong, the agent is right,
       and the demo script needs rewriting -- not the threshold.
  (H2) The centroid test cannot RESOLVE an intra-family gap: a B7-H3 patent reads
       almost exactly like a B7-H1 patent, so cosine-to-footprint is near-identical
       whether the company owns that member or not. The method is blind at this
       granularity, and no threshold rescues it.

H2 is testable: compare the cosine distribution of B7 documents BMS OWNS against
B7 documents BMS DOES NOT own. If the two distributions sit on top of each other,
the signal carries no information about ownership and the whole test is decoration.

Read-only. Loads doc_vectors mmap + graph_index. No faiss, no LLM.
"""
import sys
import numpy as np

sys.path.insert(0, "/workspace/dh_multiagent")
from agents.graph_index import GraphIndex
from discovery_hub import config

gi = GraphIndex.load(config.ARTIFACT_DIR / "graph_index.npz")
vecs = np.load(config.EMB_DIR / "doc_vectors.npy", mmap_mode="r")
import json
doc_ids = json.loads((config.EMB_DIR / "doc_ids.json").read_text())
row_of = {d: i for i, d in enumerate(doc_ids)}

# BMS portfolio across canonical variants (what the agent actually builds).
bms = gi.org_row("organization:bristol myers squibb")
variants = gi.canonical_org_rows(bms)
port = np.unique(np.concatenate([gi.portfolio(r) for r in variants]))
print(f"BMS portfolio: {len(port)} techs across {len(variants)} variants")

port_docs = [gi.doc_id_of_tech(int(r)) for r in port]
port_rows = [row_of[d] for d in port_docs if d in row_of]
P = np.asarray(vecs[sorted(port_rows)], dtype=np.float32)
print(f"portfolio docs with embeddings: {len(P)}")

from sklearn.cluster import KMeans
k = min(8, max(2, len(P) // 25))
km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit(P)
C = km.cluster_centers_
C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
print(f"{k} centroids\n")

# Find B7-family documents by title/abstract text, and split by BMS ownership.
# Titles come from the graph's technology labels -- no corpus re-read needed.
owned = set(int(r) for r in port)
b7_owned, b7_not = [], []
for row in range(gi.n_nodes):
    if gi.ntype(row) != "technology":
        continue
    lab = (gi.label(row) or "").lower()
    if "b7" not in lab:
        continue
    d = gi.doc_id_of_tech(row)
    if d not in row_of:
        continue
    (b7_owned if row in owned else b7_not).append((d, gi.label(row)))

def cos_to_footprint(docs):
    if not docs:
        return np.array([])
    V = np.asarray(vecs[[row_of[d] for d, _ in docs]], dtype=np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return (V @ C.T).max(axis=1)

so, sn = cos_to_footprint(b7_owned), cos_to_footprint(b7_not)
print(f'B7-titled docs BMS OWNS      : n={len(b7_owned):4d}  cos-to-nearest-centroid '
      f'mean={so.mean():.3f} min={so.min():.3f} max={so.max():.3f}' if len(so) else "none owned")
print(f'B7-titled docs BMS does NOT  : n={len(b7_not):4d}  cos-to-nearest-centroid '
      f'mean={sn.mean():.3f} min={sn.min():.3f} max={sn.max():.3f}' if len(sn) else "none unowned")

if len(so) and len(sn):
    thr = 0.40
    print(f"\nAt the shipped threshold {thr}:")
    print(f"  owned    flagged as 'gap': {(so < thr).sum():4d} / {len(so)}  ({100*(so<thr).mean():.1f}%)")
    print(f"  NOT owned flagged as 'gap': {(sn < thr).sum():4d} / {len(sn)}  ({100*(sn<thr).mean():.1f}%)")
    print(f"\n  If these two rates are the same, the test tells you NOTHING about ownership.")
    # Threshold sweep: is there ANY cut that separates owned from not-owned?
    print("\n  threshold sweep (can any cut separate them?)")
    print("   thr   %owned flagged   %unowned flagged   separation")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        a, b = 100 * (so < t).mean(), 100 * (sn < t).mean()
        print(f"  {t:.2f}      {a:5.1f}            {b:5.1f}          {b-a:+6.1f}")

# The spec's specific claim: B7-H3 / B7-H4 / B7-H5 are gaps BMS lacks.
print("\n=== the spec's money shot, checked against the graph ===")
for fam in ["b7-h1", "b7-h3", "b7-h4", "b7-h5", "b7h3", "b7-dc"]:
    o = [t for d, t in b7_owned if fam in t.lower()]
    n = [t for d, t in b7_not if fam in t.lower()]
    print(f"  {fam:6s}  BMS owns {len(o):3d}   others own {len(n):3d}")
