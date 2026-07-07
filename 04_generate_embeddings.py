#!/usr/bin/env python3
"""

! mock run:
python3 04_generate_embeddings.py --mock --devices a,b

! actual run:
nohup python3 04_generate_embeddings.py --devices cuda:0,cuda:1 --batch-size 64 > embed.log 2>&1 &
tail -f embed.log


04_generate_embeddings.py  --  LAYER 2, step 4 of 6.  [dual-GPU + checkpointed]

Embeds every DiscoveryDoc's embedding_text into a dense vector matrix. Drop-in
replacement for the single-pass version: SAME outputs (doc_vectors.npy + aligned
doc_ids.json), plus two robustness features for long runs on modest GPUs:

  * CHECKPOINTED -- the corpus is split into shards; each shard is embedded and
    written atomically (temp + rename). If the run dies (OOM, power, SSH drop),
    just re-run the same command: finished shards are skipped and it resumes.
  * MULTI-GPU -- one worker process per device in --devices, each pinned to its
    own GPU, draining a shared queue of pending shards. Two RTX 3060s ~halve the
    wall-clock vs one. Works for 1 device too (still checkpointed).

Outputs (unchanged, so stage 05 consumes them as-is):
  data/embeddings/doc_vectors.npy   float32 [N, dim], L2-normalized, doc-order
  data/embeddings/doc_ids.json      ordered doc_id list aligned to the rows

Usage:
  python 04_generate_embeddings.py --mock                         # plumbing test, no GPU
  python 04_generate_embeddings.py --devices cuda:0,cuda:1        # both GPUs, real model
  python 04_generate_embeddings.py --devices cuda:0 --batch-size 32   # one GPU, smaller batch
  # crashed mid-run? re-run the exact command -- it resumes from the last shard.

Note: the final concat loads all shards into RAM (~2.5 GB for 600k x 1024 f32).
At Anvil/40M scale, memmap the output instead; fine for the Drew MVP.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reduce fragmentation OOMs on 12 GB cards (the "reserved but unallocated" case).
# Set before torch/CUDA initializes; inherited by spawned workers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from discovery_hub import config
from discovery_hub.determinism import set_global_determinism
from discovery_hub.embedding import get_embedder
from discovery_hub.schema import read_docs


def _shard_path(shards_dir: Path, i: int) -> Path:
    return shards_dir / f"shard_{i:05d}.npy"


def _shard_indices(n: int, shard_size: int) -> list[int]:
    return list(range((n + shard_size - 1) // shard_size))


def _load_texts(docs_path: Path) -> list[str]:
    return [d.embedding_text for d in read_docs(docs_path)]


def _worker(device: str, mock: bool, batch_size: int, docs_path: str,
            shards_dir: Path, shard_size: int, task_q: "mp.Queue", seed: int,
            max_seq_length: int) -> None:
    """Pull shard indices off task_q, embed each, write its shard atomically."""
    set_global_determinism(seed)
    embedder = (get_embedder(mock=True) if mock
                else get_embedder(mock=False, device=device, batch_size=batch_size))
    if not mock:
        # Cap input length so per-batch memory is bounded regardless of a stray
        # very-long abstract (Qwen3 pads each batch to its longest input).
        embedder.model.max_seq_length = max_seq_length
    texts = _load_texts(Path(docs_path))
    n = len(texts)
    while True:
        i = task_q.get()
        if i is None:                       # sentinel: no more work
            break
        s, e = i * shard_size, min(i * shard_size + shard_size, n)
        t0 = time.time()
        vecs = embedder.encode_documents(texts[s:e], batch_size=batch_size).astype(np.float32)
        tmp = shards_dir / f"shard_{i:05d}.tmp"
        with open(tmp, "wb") as fh:         # file object -> np.save won't munge the name
            np.save(fh, vecs)
        os.replace(tmp, _shard_path(shards_dir, i))   # atomic: shard appears only when complete
        print(f"  [{device}] shard {i:>4} ({e - s} docs) {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="deterministic mock embedder (no GPU)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=512,
                    help="truncate inputs to this many tokens (bounds GPU memory; "
                         "512 covers almost all abstracts)")
    ap.add_argument("--devices", default="cuda:0,cuda:1",
                    help="comma-separated devices; one worker per device")
    ap.add_argument("--shard-size", type=int, default=20000,
                    help="docs per checkpoint shard (crash loses at most this many)")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep shard files after combining (default: delete)")
    args = ap.parse_args()

    config.ensure_dirs()
    docs_path = config.NORM_DIR / "docs.jsonl"
    if not docs_path.exists():
        print(f"ERROR: {docs_path} not found -- run stage 02 first.", file=sys.stderr)
        return 2
    shards_dir = config.EMB_DIR / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    ids = [d.doc_id for d in read_docs(docs_path)]      # doc order == row order
    n = len(ids)
    shards = _shard_indices(n, args.shard_size)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    print(f"Embedding {n} docs (mock={args.mock}, "
          f"model={'mock' if args.mock else config.EMBED_MODEL})")
    print(f"  {len(shards)} shard(s) x {args.shard_size}, devices={devices}")

    pending = [i for i in shards if not _shard_path(shards_dir, i).exists()]
    print(f"  {len(shards) - len(pending)} done, {len(pending)} pending")

    if pending:
        mp.set_start_method("spawn", force=True)       # required for CUDA in children
        task_q: mp.Queue = mp.Queue()
        for i in pending:
            task_q.put(i)
        for _ in devices:                              # one sentinel per worker
            task_q.put(None)
        procs = []
        for dev in devices:
            p = mp.Process(target=_worker,
                           args=(dev, args.mock, args.batch_size, str(docs_path),
                                 shards_dir, args.shard_size, task_q, config.SEED,
                                 args.max_seq_length))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        missing = [i for i in shards if not _shard_path(shards_dir, i).exists()]
        if missing:
            print(f"ERROR: {len(missing)} shard(s) did not complete (e.g. {missing[:8]}). "
                  f"A worker likely crashed -- if it was CUDA OOM, lower --batch-size and "
                  f"re-run (it resumes).", file=sys.stderr)
            return 1

    # --- combine shards in order -> single aligned matrix ---
    print("combining shards ...")
    mats = [np.load(_shard_path(shards_dir, i)) for i in shards]
    vectors = np.vstack(mats) if mats else np.zeros((0, config.EMBED_DIM), np.float32)
    if vectors.shape[0] != n:
        print(f"ERROR: row count {vectors.shape[0]} != doc count {n}; not writing.",
              file=sys.stderr)
        return 1

    np.save(config.EMB_DIR / "doc_vectors.npy", vectors.astype(np.float32))
    with (config.EMB_DIR / "doc_ids.json").open("w") as fh:
        json.dump(ids, fh)
    print(f"Wrote vectors {vectors.shape} -> {config.EMB_DIR / 'doc_vectors.npy'}")

    if not args.keep_shards:
        for i in shards:
            _shard_path(shards_dir, i).unlink()
        try:
            shards_dir.rmdir()
        except OSError:
            pass
        print("  removed shard dir (pass --keep-shards to retain for resume)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
