#!/usr/bin/env python3
"""Produce docs_v2.jsonl: unique doc_ids, exact-duplicate rows dropped.

The v1 corpus has 603,369 rows but 600,738 unique doc_ids. All 2,631 extra rows are
SBIR: 01_7_convert_sbir.py:67-72 uses the raw award/contract id as the doc_id, and
those recur across award years and phases (sbir:PHS2001-2 covers 144 different
companies' awards).

Scheme: colliding SBIR ids become  sbir:{award}:{sha256(record-without-doc_id)[:8]}
  - content-addressed, so it is reproducible from this file alone with no join back
    to sbir_bulk.json (that join would key on the ambiguous id and be circular);
  - rows whose canonical record is identical collapse to the same new id and the
    extras are dropped -- they are the same document, not two awards;
  - non-colliding ids and every non-SBIR row are left byte-identical.

Consumers all use doc_id.split(":", 1), so a third :-component is safe.

Writes:
  docs_v2.jsonl            retained rows, new ids, v1 relative order preserved
  docs_v2_keep_index.json  v1 row ordinal of each retained row -- REQUIRED to
                           row-gather v2 vectors out of the v1 .npy without re-embedding
  docs_v2_id_map.json      old_doc_id -> [new_doc_id, ...] for the relabelled groups
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def rec_hash(o: dict) -> str:
    d = {k: v for k, v in o.items() if k != "doc_id"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="normalized/docs.jsonl")
    ap.add_argument("--out-docs", default="normalized/docs_v2.jsonl")
    ap.add_argument("--out-keep-index", default="normalized/docs_v2_keep_index.json")
    ap.add_argument("--out-id-map", default="normalized/docs_v2_id_map.json")
    ap.add_argument("--hash-chars", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="without this, dry-run only")
    args = ap.parse_args()

    src = Path(args.docs)

    # pass 1 -- find colliding ids
    counts: Counter[str] = Counter()
    with src.open() as fh:
        for line in fh:
            counts[json.loads(line)["doc_id"]] += 1
    collide = {k for k, v in counts.items() if v > 1}
    print(f"[relabel] v1 rows={sum(counts.values())} unique={len(counts)} colliding_ids={len(collide)}")
    if any(not k.startswith("sbir:") for k in collide):
        raise SystemExit("ERROR: a non-SBIR id collides. The scheme assumes SBIR-only. Stop.")

    # pass 2 -- assign, dedupe, emit
    keep_index: list[int] = []
    id_map: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    dropped = 0
    relabelled = 0
    out_lines: list[str] = []

    with src.open() as fh:
        for i, line in enumerate(fh):
            o = json.loads(line)
            did = o["doc_id"]
            if did not in collide:
                keep_index.append(i)
                out_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            nid = f"{did}:{rec_hash(o)[: args.hash_chars]}"
            if nid in seen:          # identical record already emitted -> same document
                dropped += 1
                continue
            seen.add(nid)
            o["doc_id"] = nid
            id_map[did].append(nid)
            relabelled += 1
            keep_index.append(i)
            out_lines.append(json.dumps(o, ensure_ascii=False) + "\n")

    final_ids = [json.loads(l)["doc_id"] for l in out_lines]
    dup = {k: v for k, v in Counter(final_ids).items() if v > 1}

    print(f"[relabel] dropped_exact_duplicates={dropped}")
    print(f"[relabel] relabelled_rows={relabelled}")
    print(f"[relabel] v2 rows={len(out_lines)} unique_ids={len(set(final_ids))}")
    print(f"[relabel] REMAINING COLLISIONS={len(dup)}")
    if dup:
        raise SystemExit(f"ERROR: v2 still has collisions: {list(dup.items())[:5]}")
    assert len(keep_index) == len(out_lines) == len(final_ids)

    if not args.write:
        print("[relabel] DRY RUN -- pass --write to emit")
        return 0

    Path(args.out_docs).write_text("".join(out_lines))
    Path(args.out_keep_index).write_text(json.dumps(keep_index))
    Path(args.out_id_map).write_text(json.dumps(dict(id_map), indent=1))
    print(f"[relabel] wrote {args.out_docs} ({len(out_lines)} rows)")
    print(f"[relabel] wrote {args.out_keep_index} ({len(keep_index)} ordinals)")
    print(f"[relabel] wrote {args.out_id_map} ({len(id_map)} remapped ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
