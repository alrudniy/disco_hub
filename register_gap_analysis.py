import json, re, sys, numpy as np
from pathlib import Path
DOCS  = sys.argv[1] if len(sys.argv) > 1 else "/workspace/dh_data/normalized/docs.jsonl"
QRELS = sys.argv[2] if len(sys.argv) > 2 else "/workspace/dh_data/pipeline2/qrels/qrels_exact_origin_v1.jsonl"
PQ    = sys.argv[3] if len(sys.argv) > 3 else "/workspace/per_query_r10.json"
STOP = set("the a an of for and or to in with as by from thereof use uses using method methods "
           "composition compositions available licensing novel new therapy treatment".split())
def toks(s): return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in STOP}
text = {}
for l in Path(DOCS).open():
    if not l.strip(): continue
    d = json.loads(l)
    text[d["doc_id"]] = (d.get("title") or "") + " " + (d.get("embedding_text") or d.get("abstract") or "")[:1500]
qr = [json.loads(l) for l in Path(QRELS).open() if l.strip()]
h  = {k: np.array(v) for k, v in json.load(open(PQ)).items()}
ov = np.array([len(toks(q["query"]) & toks(text.get(q["relevant_doc_ids"][0], ""))) / max(len(toks(q["query"])),1)
               for q in qr])
med = np.median(ov); LO, HI = ov < med, ov >= med
rng = np.random.default_rng(0)
print(f"n={len(ov)}  median query<->origin overlap={med:.3f}  LOW n={LO.sum()}  HIGH n={HI.sum()}\n")
print(f"{'model':24s} {'LOW':>8s} {'HIGH':>8s} {'ratio':>7s}")
for m in ["Z_baseline_untrained","A_mnrl_control","A_fixed"]:
    print(f"{m:24s} {h[m][LO].mean():8.4f} {h[m][HI].mean():8.4f} {h[m][HI].mean()/h[m][LO].mean():7.2f}")
print("\nDiD vs Z (HIGH gain minus LOW gain), 20k paired bootstrap:")
for x in ["A_mnrl_control","A_fixed"]:
    dh_ = h[x][HI] - h["Z_baseline_untrained"][HI]; dl_ = h[x][LO] - h["Z_baseline_untrained"][LO]
    did = (dh_[rng.integers(0,len(dh_),(20000,len(dh_)))].mean(axis=1)
         - dl_[rng.integers(0,len(dl_),(20000,len(dl_)))].mean(axis=1))
    lo, hi = np.percentile(did, [2.5, 97.5])
    print(f"  {x:20s} {did.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if lo>0 or hi<0 else 'ns'}")
