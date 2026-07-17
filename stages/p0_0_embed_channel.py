#!/usr/bin/env python3
"""p0_0 -- embed the corpus with ONE channel model -> (doc_vectors.npy, doc_ids.json).

THE BLOCKING PREREQUISITE. The cascade unions the untouched Qwen3-Embedding-0.6B with the
fine-tuned models, and every channel needs its OWN vectors: embeddings are not
interchangeable across models (different dims, different spaces). The untouched-0.6B
vectors do not exist -- pipeline_1 never built them, because that model was never
interesting until it turned out to be the best low-overlap matcher in the whole eval
(LOW R@10 0.2556 vs 0.1667 for the best trained arm).

Without this stage, p0_1 / p1_4 / 07 all warn and fall back to the fine-tuned-only pool,
which is the exact configuration measured to LOSE 0.089 R@10 on low-overlap queries.

INVARIANT (README, carried from pipeline_1): docs.jsonl line order == vector row order ==
index row order. This stage reads docs.jsonl ONCE and writes ids in that same order, then
records a doc_order_checksum in the manifest so a later mismatch is detectable rather
than silently wrong.

Usage -- the untouched 0.6B (~45 min on a warm GPU, 600,738 docs):

    python stages/p0_0_embed_channel.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --out-dir $DH2_ROOT/vectors/base06b

Then point the cascade at it:

    export DH_CASCADE_VECTORS_BASE_06B=$DH2_ROOT/vectors/base06b/doc_vectors.npy
    export DH_CASCADE_IDS_BASE_06B=$DH2_ROOT/vectors/base06b/doc_ids.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dh2 import config2 as C
from dh2 import manifest as M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model dir or HF id, e.g. Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out-dir", required=True,
                    help="destination for doc_vectors.npy + doc_ids.json")
    ap.add_argument("--docs", default=str(C.PROD_DOCS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=0,
                    help="override the model's default (0 = leave alone)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing vectors")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vpath, ipath = out / "doc_vectors.npy", out / "doc_ids.json"

    # ~45 min of GPU is not something to redo by accident.
    if vpath.exists() and not args.force:
        print(f"[p0_0] {vpath} already exists. Use --force to re-embed.")
        ids = json.loads(ipath.read_text()) if ipath.exists() else []
        v = np.load(vpath, mmap_mode="r")
        print(f"[p0_0] existing: {v.shape[0]} vectors x {v.shape[1]} dims, {len(ids)} ids")
        return 0

    from sentence_transformers import SentenceTransformer
    from discovery_hub.schema import read_docs

    print(f"[p0_0] reading {args.docs} ...")
    docs = list(read_docs(Path(args.docs)))
    # Read order IS the contract. Do not sort, filter, or dedupe below this line.
    texts = [d.embedding_text or f"{d.title}\n{d.abstract}" for d in docs]
    ids = [d.doc_id for d in docs]
    print(f"[p0_0] {len(docs)} docs")

    print(f"[p0_0] loading {args.model} ...")
    m = SentenceTransformer(args.model, device=args.device)
    if args.max_seq_length:
        m.max_seq_length = args.max_seq_length
    print(f"[p0_0] max_seq_length={m.max_seq_length}")

    t0 = time.perf_counter()
    vecs = m.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                    show_progress_bar=True, convert_to_numpy=True)
    mins = (time.perf_counter() - t0) / 60
    vecs = vecs.astype("float32")

    # Sentence-Transformers applies normalize_embeddings inside the model's COMPUTE dtype.
    # Checkpoints that run in reduced precision (bf16/fp16) therefore return rows whose
    # norms land at 1 +/- ~4e-3 once cast up to float32 -- enough to violate the unit-norm
    # contract DenseRetriever relies on (IndexFlatIP == cosine), which the assert below
    # rightly catches. Re-normalize at float32 rather than loosening the tolerance: cosine
    # is scale-invariant per row, so this changes NO ranking; it only makes the contract
    # exactly true. (The stock 0.6B lands at 0.9998 and passes either way; qwen3-dh-ft
    # does not -- min 0.9980 / max 1.0037.)
    row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(row_norms, 1e-12)

    assert vecs.shape[0] == len(ids), \
        f"row/id mismatch: {vecs.shape[0]} vectors vs {len(ids)} ids"
    # DenseRetriever assumes unit-norm vectors (IndexFlatIP == cosine). Check, don't hope.
    norms = np.linalg.norm(vecs[: min(1000, len(vecs))], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), \
        f"vectors are not unit-norm (min {norms.min():.4f}, max {norms.max():.4f})"

    np.save(vpath, vecs)
    ipath.write_text(json.dumps(ids))

    checksum = M.doc_order_checksum(ids)
    try:
        M.write_manifest(out / "manifest.json", artifact_kind="doc_vectors",
                         base_model=args.model, doc_ids=ids,
                         embedding_dim=int(vecs.shape[1]),
                         max_seq_length=int(m.max_seq_length),
                         repo_dir=str(Path(__file__).resolve().parents[1]),
                         extra={"n_docs": len(ids), "docs_path": str(args.docs),
                                "channel": "embedding", "normalized": True})
    except Exception as e:                                      # noqa: BLE001
        # A manifest is provenance, not payload -- never lose 45 min of GPU over it.
        print(f"[p0_0] WARNING: manifest not written ({e})")

    print(f"\n[p0_0] wrote {vecs.shape[0]} x {vecs.shape[1]} -> {vpath}  ({mins:.1f} min)")
    print(f"[p0_0] doc_order_checksum: {checksum}")
    print(f"[p0_0] ids -> {ipath}")
    print(f"\n[p0_0] point the cascade at it:")
    print(f"    export DH_CASCADE_VECTORS_BASE_06B={vpath}")
    print(f"    export DH_CASCADE_IDS_BASE_06B={ipath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
