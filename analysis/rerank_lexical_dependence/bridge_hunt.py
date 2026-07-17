"""
Does this encoder bridge a SYNONYM / REGISTER gap? Measured, not asserted.

The spec's demo query 1 claims retrieval surfaces "B7-H1, a novel immunoregulatory
molecule" for a PD-L1 query, "which no keyword search finds". Measured: that patent
ranks 7,585 of 603,369 (cosine 0.234). So the claim is false for that pair. The
question this script answers is whether it is false for EVERY pair, or whether there
is a real one to demo.

THE TEST, and why it is fair:
  For a pair (modern, legacy) -- e.g. PD-L1 / B7-H1 -- take only documents whose text
  contains the LEGACY term and NOT the modern one. Query with the MODERN term. Because
  the modern term is absent from those documents BY CONSTRUCTION, a keyword search on
  it cannot return them at any rank. So if dense retrieval puts one in the top-k, that
  is a genuine semantic bridge and not lexical overlap leaking through.

  A pair only counts if such documents EXIST (else the test is vacuous), so the script
  reports the candidate pool size for every pair and never scores an empty one.

Reports the rank of the best legacy-only document per pair. Rank 1-10 = a demoable
bridge. Rank in the thousands = the encoder treats the synonyms as unrelated.
"""
import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/dh_multiagent")
from discovery_hub import config
from discovery_hub.embedding import get_embedder

# (label, modern term used in the QUERY, [legacy/alternative terms that must appear],
#  natural research-interest phrasing built around the MODERN term only)
PAIRS = [
    ("PD-L1 / B7-H1", "pd-l1", ["b7-h1", "b7h1"],
     "monoclonal antibody targeting PD-L1 for oncology"),
    ("PD-L1 / CD274", "pd-l1", ["cd274"],
     "monoclonal antibody targeting PD-L1 for oncology"),
    ("PD-1 / PDCD1", "pd-1", ["pdcd1"],
     "antibody blocking PD-1 for cancer immunotherapy"),
    ("PD-1 / CD279", "pd-1", ["cd279"],
     "antibody blocking PD-1 for cancer immunotherapy"),
    ("HER2 / ERBB2", "her2", ["erbb2", "erbb-2"],
     "HER2 targeted therapy for breast cancer"),
    ("HER2 / neu", "her2", ["her-2/neu", "c-erbb-2"],
     "HER2 targeted therapy for breast cancer"),
    ("EGFR / ErbB-1", "egfr", ["erbb1", "erbb-1", "her1"],
     "EGFR inhibitor for non-small cell lung cancer"),
    ("CTLA-4 / CD152", "ctla-4", ["cd152"],
     "CTLA-4 blockade for melanoma"),
    ("4-1BB / CD137", "4-1bb", ["cd137", "tnfrsf9"],
     "4-1BB agonist for T cell costimulation"),
    ("OX40 / CD134", "ox40", ["cd134", "tnfrsf4"],
     "OX40 agonist antibody for tumor immunotherapy"),
    ("LAG-3 / CD223", "lag-3", ["cd223"],
     "LAG-3 checkpoint inhibitor for cancer"),
    ("TIM-3 / HAVCR2", "tim-3", ["havcr2"],
     "TIM-3 checkpoint blockade for cancer"),
    ("B7-H3 / CD276", "b7-h3", ["cd276"],
     "B7-H3 targeted antibody for solid tumors"),
    ("B7-H4 / VTCN1", "b7-h4", ["vtcn1"],
     "B7-H4 targeted immunotherapy"),
    ("ICOS / CD278", "icos", ["cd278"],
     "ICOS agonist for T cell activation"),
    ("VEGF / vascular permeability factor", "vegf", ["vascular permeability factor"],
     "VEGF inhibitor for angiogenesis in cancer"),
    ("TNF-alpha / cachectin", "tnf-alpha", ["cachectin"],
     "TNF-alpha inhibitor for inflammatory disease"),
    ("IL-2 / T cell growth factor", "il-2", ["t cell growth factor",
                                             "t-cell growth factor"],
     "IL-2 therapy for immune activation"),
    ("trastuzumab / Herceptin", "trastuzumab", ["herceptin"],
     "trastuzumab for HER2 positive breast cancer"),
    ("pembrolizumab / Keytruda", "pembrolizumab", ["keytruda"],
     "pembrolizumab for non-small cell lung cancer"),
    ("pembrolizumab / MK-3475", "pembrolizumab", ["mk-3475"],
     "pembrolizumab for non-small cell lung cancer"),
    ("nivolumab / Opdivo", "nivolumab", ["opdivo"],
     "nivolumab checkpoint inhibitor therapy"),
    ("nivolumab / BMS-936558", "nivolumab", ["bms-936558"],
     "nivolumab checkpoint inhibitor therapy"),
    ("semaglutide / Ozempic", "semaglutide", ["ozempic", "wegovy"],
     "semaglutide for type 2 diabetes and weight loss"),
    ("GLP-1 / glucagon-like peptide", "glp-1", ["glucagon-like peptide-1",
                                                "glucagon like peptide 1"],
     "GLP-1 receptor agonist for metabolic disease"),
    ("aspirin / acetylsalicylic acid", "aspirin", ["acetylsalicylic acid"],
     "aspirin for cardiovascular prevention"),
    ("statin / HMG-CoA reductase inhibitor", "statin", ["hmg-coa reductase inhibitor"],
     "statin therapy for lowering cholesterol"),
    ("CAR-T / chimeric antigen receptor", "car-t", ["chimeric antigen receptor"],
     "CAR-T cell therapy for leukemia"),
]

def _squash(s: str) -> str:
    """Lowercase and drop every separator: 'PD-L1' / 'PDL1' / 'PD L1' -> 'pdl1'."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


print("loading corpus text ...", flush=True)
titles, texts, doc_ids = {}, {}, []
with open(config.NORM_DIR / "docs.jsonl") as fh:
    for line in fh:
        d = json.loads(line)
        did = d["doc_id"]
        doc_ids.append(did)
        titles[did] = d.get("title") or ""
        # SEPARATOR-BLIND. The first pass of this experiment used a plain lowercase
        # substring test and it leaked: "PD-1 / CD279" scored rank 3 on
        # "Anti-PD1 antibodies", and "B7-H3 / CD276" scored rank 17 on "Anti-CD276
        # antibodies (B7H3)". Both documents spell the MODERN term without the hyphen,
        # so a keyword search finds them trivially and neither is a semantic bridge.
        # Squashing every non-alphanumeric character makes pd-1 / PD1 / "PD 1" one
        # token, so the exclusion means what the experiment claims it means.
        texts[did] = _squash((d.get("title") or "") + " " + (d.get("abstract") or ""))
print(f"  {len(doc_ids)} docs", flush=True)

emb_ids = json.loads((config.EMB_DIR / "doc_ids.json").read_text())
row_of = {d: i for i, d in enumerate(emb_ids)}

print("loading vectors to GPU ...", flush=True)
V = np.load(config.EMB_DIR / "doc_vectors.npy", mmap_mode="r")
Vg = torch.from_numpy(np.asarray(V, dtype=np.float32)).cuda()
print(f"  {tuple(Vg.shape)} on {Vg.device}", flush=True)

emb = get_embedder(mock=False, device="cuda")

print(f"\n{'pair':38s} {'pool':>6s} {'best rank':>10s} {'cos':>7s}  verdict")
print("-" * 92)
results = []
for label, modern, legacies, query in PAIRS:
    # Legacy-only documents: contain a legacy term, never the modern one. A keyword
    # search for `modern` cannot return these at any rank -- that is the control.
    m = _squash(modern)
    ls = [_squash(l) for l in legacies]
    pool = [d for d in doc_ids
            if m not in texts[d] and any(l in texts[d] for l in ls)]
    if not pool:
        print(f"{label:38s} {0:>6d} {'--':>10s} {'--':>7s}  no legacy-only docs (vacuous)")
        continue

    qv = torch.from_numpy(emb.encode_queries([query])[0].astype(np.float32)).cuda()
    sims = (Vg @ qv)
    rows = [row_of[d] for d in pool if d in row_of]
    if not rows:
        print(f"{label:38s} {len(pool):>6d} {'--':>10s} {'--':>7s}  no embeddings")
        continue
    pool_sims = sims[torch.tensor(rows, device=sims.device)]
    best_i = int(torch.argmax(pool_sims))
    best_sim = float(pool_sims[best_i])
    best_doc = [d for d in pool if d in row_of][best_i]
    rank = int((sims > best_sim).sum()) + 1

    verdict = ("BRIDGE (top-10)" if rank <= 10 else
               "top-50" if rank <= 50 else
               "top-200" if rank <= 200 else "no bridge")
    print(f"{label:38s} {len(pool):>6d} {rank:>10d} {best_sim:>7.3f}  {verdict}")
    results.append((rank, label, best_doc, titles[best_doc], best_sim, query, len(pool)))

print("\n" + "=" * 92)
print("BEST BRIDGES (a keyword search for the queried term CANNOT return these):")
for rank, label, doc, title, sim, query, pool in sorted(results)[:6]:
    print(f"\n  rank {rank:>6d} of 603,369   cos {sim:.3f}   pool {pool}   [{label}]")
    print(f"    query : {query!r}")
    print(f"    doc   : {doc}  {title[:80]!r}")
