#!/usr/bin/env python3
"""
01_2_filter_openalex.py  --  bridge between the OpenAlex S3 snapshot and stage 02.

Streams the gzipped works partitions downloaded by 01_1_download_bulk.sh, keeps
only medicine/biology works, and writes them as parse-ready JSONL at
$DH_DATA_ROOT/raw/openalex.jsonl (the shape 02_parse_normalize.parse_openalex
consumes -- the raw OpenAlex work object, with _retrieved_date stamped).

Why this exists: the 2026 snapshot has ~463M works (~1.6 TB unzipped). Drew cannot
embed that. This cuts it to the medicine+biology slice (and optionally caps it for
an MVP) so stage 04 only ever sees a Drew-sized corpus.

Robust matching: OpenAlex is migrating from `concepts` (deprecated) to `topics`, so
a work is kept if EITHER it carries a medicine/biology concept (>= --min-score) OR
its primary/any topic sits in a Health/Life Sciences domain. Matching on concepts
alone would silently drop newer works that only have topics.

Streaming + resumable: each input partition is filtered into its own shard under
--shard-dir via a temp+rename (atomic), so an interrupted run just re-runs the
unfinished partitions -- no duplicates, no re-doing finished work. Shards are then
concatenated into --out.

Examples:
  # full medicine/biology subset
  python 01_2_filter_openalex.py
  # MVP: stop after 2M kept works
  python 01_2_filter_openalex.py --max 2000000 --require-abstract
  # quick smoke test on the first 2 partitions
  python 01_2_filter_openalex.py --limit-files 2 --keep-shards
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# Legacy `concepts` ids (URL tail) and current `topics` domains that mark a work
# as in-scope. Either signal is sufficient.
DEFAULT_CONCEPTS = "C71924100,C86803240"          # Medicine, Biology
DEFAULT_DOMAINS = "Health Sciences,Life Sciences"  # OpenAlex topic domains


def _data_root() -> Path:
    return Path(os.environ.get("DH_DATA_ROOT", "./data")).resolve()


def is_in_scope(work: dict, concept_ids: set[str], min_score: float,
                domains: set[str]) -> bool:
    """True if the work is medicine/biology by concept OR by topic domain."""
    # 1. legacy concepts (id is a URL like https://openalex.org/C71924100)
    for c in work.get("concepts") or []:
        cid = (c.get("id") or "").rsplit("/", 1)[-1]
        if cid in concept_ids and (c.get("score") or 0.0) >= min_score:
            return True
    # 2. current topics: check primary_topic, then all topics, for a domain hit
    pt = work.get("primary_topic") or {}
    if ((pt.get("domain") or {}).get("display_name") or "") in domains:
        return True
    for t in work.get("topics") or []:
        if ((t.get("domain") or {}).get("display_name") or "") in domains:
            return True
    return False


def _has_abstract(work: dict) -> bool:
    inv = work.get("abstract_inverted_index")
    return bool(inv)


def _shard_name(input_path: Path, snapshot_dir: Path) -> str:
    """Stable, filesystem-safe shard name from an input partition path."""
    rel = input_path.relative_to(snapshot_dir)
    return str(rel).replace(os.sep, "__").replace("=", "-") + ".jsonl"


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def filter_partition(input_path: Path, shard_path: Path, concept_ids: set[str],
                     min_score: float, domains: set[str], require_abstract: bool,
                     retrieved: str, remaining: int | None) -> tuple[int, int, int]:
    """
    Filter one .gz partition into shard_path (atomic via .tmp + rename).
    `remaining` caps how many kept records to write (None = unlimited).
    Returns (scanned, kept, bad_lines).
    """
    tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    scanned = kept = bad = 0
    with gzip.open(input_path, "rt", encoding="utf-8") as fin, \
            open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not is_in_scope(work, concept_ids, min_score, domains):
                continue
            if require_abstract and not _has_abstract(work):
                continue
            work["_retrieved_date"] = retrieved
            fout.write(json.dumps(work, ensure_ascii=False) + "\n")
            kept += 1
            if remaining is not None and kept >= remaining:
                break
    os.replace(tmp, shard_path)   # atomic: shard only appears once complete
    return scanned, kept, bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", default=None,
                    help="dir holding the works partitions "
                         "(default $DH_DATA_ROOT/openalex_snapshot/works)")
    ap.add_argument("--out", default=None,
                    help="final JSONL (default $DH_DATA_ROOT/raw/openalex.jsonl)")
    ap.add_argument("--shard-dir", default=None,
                    help="per-partition shards (default "
                         "$DH_DATA_ROOT/openalex_filtered)")
    ap.add_argument("--concepts", default=DEFAULT_CONCEPTS,
                    help="comma-separated OpenAlex concept ids to keep")
    ap.add_argument("--domains", default=DEFAULT_DOMAINS,
                    help="comma-separated topic domains to keep")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="min concept score to count a concept match (default 0.3)")
    ap.add_argument("--max", type=int, default=0,
                    help="stop after this many kept works (0 = no cap)")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="only process the first N partitions (0 = all)")
    ap.add_argument("--require-abstract", action="store_true",
                    help="drop works with no abstract (recommended for embedding)")
    ap.add_argument("--no-combine", action="store_true",
                    help="write shards only; skip concatenation into --out")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep the shard dir after combining (default: delete)")
    ap.add_argument("--progress-every", type=int, default=2_000_000,
                    help="log progress every N scanned works")
    args = ap.parse_args()

    root = _data_root()
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else \
        root / "openalex_snapshot" / "works"
    out_path = Path(args.out) if args.out else root / "raw" / "openalex.jsonl"
    shard_dir = Path(args.shard_dir) if args.shard_dir else \
        root / "openalex_filtered"

    if not snapshot_dir.exists():
        print(f"ERROR: snapshot dir not found: {snapshot_dir}\n"
              f"  Did 01_1_download_bulk.sh finish the OpenAlex sync? Expected the "
              f"works partitions there.", file=sys.stderr)
        return 2

    concept_ids = {c.strip() for c in args.concepts.split(",") if c.strip()}
    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    retrieved = date.today().isoformat()
    cap = args.max or None

    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # input partitions (recursive: handles updated_date=*/part_*.gz)
    files = sorted(snapshot_dir.rglob("*.gz"))
    if args.limit_files:
        files = files[:args.limit_files]
    if not files:
        print(f"ERROR: no .gz partitions under {snapshot_dir}", file=sys.stderr)
        return 2
    print(f"OpenAlex filter: {len(files)} partition(s) under {snapshot_dir}")
    print(f"  keep concepts={sorted(concept_ids)} domains={sorted(domains)} "
          f"min_score={args.min_score} require_abstract={args.require_abstract} "
          f"max={args.max or 'none'}")

    # resume: pre-existing shards already contribute to the cap
    kept_total = 0
    for f in files:
        sp = shard_dir / _shard_name(f, snapshot_dir)
        if sp.exists():
            kept_total += _count_lines(sp)

    scanned_total = bad_total = 0
    t0 = time.time()
    next_log = args.progress_every
    for i, f in enumerate(files, 1):
        shard_path = shard_dir / _shard_name(f, snapshot_dir)
        if shard_path.exists():
            continue                      # already done on a previous run
        if cap is not None and kept_total >= cap:
            break
        remaining = (cap - kept_total) if cap is not None else None
        s, k, b = filter_partition(f, shard_path, concept_ids, args.min_score,
                                   domains, args.require_abstract, retrieved,
                                   remaining)
        scanned_total += s
        kept_total += k
        bad_total += b
        if scanned_total >= next_log:
            rate = scanned_total / max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(files)}] scanned={scanned_total:,} "
                  f"kept={kept_total:,} ({rate:,.0f}/s)")
            next_log += args.progress_every

    print(f"\nfiltered: scanned={scanned_total:,} kept={kept_total:,} "
          f"bad_lines={bad_total:,} shards={shard_dir}")

    if args.no_combine:
        print("  (--no-combine: shards written, not concatenated)")
        return 0

    # concatenate shards -> single parse-ready JSONL
    shards = sorted(shard_dir.glob("*.jsonl"))
    print(f"combining {len(shards)} shard(s) -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as out:
        for sp in shards:
            with open(sp, "r", encoding="utf-8") as fh:
                for line in fh:
                    out.write(line)
    size_gb = out_path.stat().st_size / 1e9
    print(f"  wrote {out_path} ({size_gb:.2f} GB)")

    if not args.keep_shards:
        for sp in shards:
            sp.unlink()
        try:
            shard_dir.rmdir()
        except OSError:
            pass
        print("  removed shard dir (pass --keep-shards to retain)")

    print(f"\nDONE. {kept_total:,} works -> {out_path}. Stage 02 can parse it with "
          f"the same DH_DATA_ROOT set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
