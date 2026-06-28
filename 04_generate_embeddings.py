#!/usr/bin/env python3
"""
04_generate_embeddings.py  --  LAYER 2 (Retrieval & Ranking), step 4 of 6.

Embeds every DiscoveryDoc's embedding_text into a dense vector matrix.

  TARGET: Anvil A100/H100 batch job at full scale. Embedding ~40-50M abstracts
          with Qwen3-Embedding-0.6B is only ~30-100 A100-hours -- a single batch
          job. The resulting doc_vectors.npy + the index from step 05 are then
          shipped to Drew for serving.

Determinism note: on a FIXED stack (same GPU + library + batch size) embeddings
are reproducible to tiny cosine drift; the mock embedder is bit-identical across
machines. 09_stability_harness.py measures the real drift.

Outputs:
  data/embeddings/doc_vectors.npy   float32 [N, dim], L2-normalized
  data/embeddings/doc_ids.json      ordered list of doc_id aligned to the rows

Usage:
  python 04_generate_embeddings.py --mock
  python 04_generate_embeddings.py --batch-size 64      # real Qwen3 on GPU
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder
from discovery_hub.schema import read_docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="deterministic mock embedder")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu (real mode)")
    args = ap.parse_args()

    set_global_determinism(config.SEED)
    config.ensure_dirs()

    docs = list(read_docs(config.NORM_DIR / "docs.jsonl"))
    texts = [d.embedding_text for d in docs]
    ids = [d.doc_id for d in docs]
    print(f"Embedding {len(texts)} docs (mock={args.mock}, model="
          f"{'mock' if args.mock else config.EMBED_MODEL})")

    embedder = get_embedder(mock=args.mock, device=args.device) if not args.mock \
        else get_embedder(mock=True)
    # Encode in chunks so memory stays flat at scale.
    chunks = []
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        chunks.append(embedder.encode_documents(texts[i:i + bs]))
    vectors = np.vstack(chunks) if chunks else np.zeros((0, embedder.dim), np.float32)

    np.save(config.EMB_DIR / "doc_vectors.npy", vectors.astype(np.float32))
    with (config.EMB_DIR / "doc_ids.json").open("w") as fh:
        json.dump(ids, fh)
    print(f"Wrote vectors {vectors.shape} -> {config.EMB_DIR / 'doc_vectors.npy'}")
    # Quick reproducibility self-check: re-encode the first 8 and compare.
    if texts:
        again = embedder.encode_documents(texts[:8])
        drift = float(np.max(np.abs(again - vectors[:8])))
        print(f"  self-check max abs drift on re-encode of 8 docs: {drift:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
