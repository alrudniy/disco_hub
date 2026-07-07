#!/usr/bin/env python3
"""
01_4_fetch_uspto_bigquery.py  --  pharma patents WITH abstracts via Google Patents.

The ODP file-wrapper API returns no abstract, so we pull USPTO patents from the
Google Patents public dataset on BigQuery (`patents-public-data.patents.publications`),
which carries title + abstract + CPC + inventors + assignees. Filters to US pharma
(CPC A61K / A61P), keeps only records that actually have an English abstract, and
writes `parse_uspto`-ready rows to $DH_DATA_ROOT/raw/uspto.jsonl.

SETUP (one-time):
  1. A Google Cloud project with billing enabled (BigQuery's first 1 TB scanned per
     month is free; this query is a one-off).
  2. A service account with role "BigQuery Job User" (+ "BigQuery User"); download
     its JSON key. On Drew:
        export GOOGLE_APPLICATION_CREDENTIALS=key-google-cloud-big-query.json
        export GOOGLE_CLOUD_PROJECT=summer-dssi
  3. pip install google-cloud-bigquery

ALWAYS dry-run first to see how many bytes the query will scan (that's what you're
billed on):
  python 01_4_fetch_uspto_bigquery.py --dry-run
Then run for real:
  python 01_4_fetch_uspto_bigquery.py                    # all US A61K/A61P granted
  
above is the defaults, no filters. Every US granted patent (kind code B%) 
whose CPC classification includes A61K or A61P (medicinal preparations /
 therapeutic activity) and that has an English abstract, across all years. 
 This is the broadest pull — likely a few hundred thousand patents. 
 A61K/A61P is the pharma filter; "granted" means issued patents, not applications.


  python 01_4_fetch_uspto_bigquery.py --max 20000        # MVP slice
  python 01_4_fetch_uspto_bigquery.py --since 2010 --include-pregrant
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

TABLE = "patents-public-data.patents.publications"
BYTES_PER_TIB = 2 ** 40
USD_PER_TIB = 6.25          # BigQuery on-demand pricing; first 1 TiB/month is free


def _data_root() -> Path:
    return Path(os.environ.get("DH_DATA_ROOT", "./data")).resolve()


def build_sql(cpc_prefixes: list[str], granted_only: bool, since_year: int | None,
              max_rows: int | None) -> str:
    # prefixes are validated in main(); safe to inline into LIKE clauses.
    like = " OR ".join(f"c.code LIKE '{p}%'" for p in cpc_prefixes)
    where = ["country_code = 'US'",
             f"EXISTS (SELECT 1 FROM UNNEST(cpc) c WHERE {like})"]
    if granted_only:
        where.append("kind_code LIKE 'B%'")          # granted utility (B1/B2)
    if since_year:
        where.append(f"filing_date >= {since_year}0101")   # INT64 YYYYMMDD
    where.append("(SELECT a.text FROM UNNEST(abstract_localized) a "
                 "WHERE a.language='en' LIMIT 1) IS NOT NULL")
    where_sql = "\n  AND ".join(where)
    limit_sql = f"\nLIMIT {max_rows}" if max_rows else ""
    return f"""
SELECT
  publication_number,
  (SELECT t.text FROM UNNEST(title_localized) t WHERE t.language='en' LIMIT 1) AS title,
  (SELECT a.text FROM UNNEST(abstract_localized) a WHERE a.language='en' LIMIT 1) AS abstract,
  ARRAY(SELECT c.code FROM UNNEST(cpc) c) AS cpc_codes,
  ARRAY(SELECT c.code FROM UNNEST(cpc) c WHERE {like}) AS pharma_cpc,
  ARRAY(SELECT ih.name FROM UNNEST(inventor_harmonized) ih WHERE ih.name IS NOT NULL) AS inventors,
  ARRAY(SELECT ah.name FROM UNNEST(assignee_harmonized) ah WHERE ah.name IS NOT NULL) AS assignees
FROM `{TABLE}`
WHERE {where_sql}{limit_sql}
"""


def row_to_record(row, retrieved: str) -> dict:
    """Map a BigQuery result row into the flat shape 02_parse_normalize.parse_uspto reads."""
    pub = (row["publication_number"] or "")
    pid = pub.replace("-", "")                       # US-9876543-B2 -> US9876543B2
    pharma_cpc = list(row["pharma_cpc"] or [])
    subclasses = sorted({c[:4] for c in pharma_cpc if len(c) >= 4})  # -> A61K, A61P
    return {
        "patent_id": pid,
        "patent_title": row["title"] or "",
        "patent_abstract": row["abstract"] or "",
        "inventors": [n for n in (row["inventors"] or []) if n],
        "assignees": [n for n in (row["assignees"] or []) if n],
        "cpc_codes": list(row["cpc_codes"] or []),
        "pharma_keywords": subclasses,
        "_source_url": f"https://patents.google.com/patent/{pid}",
        "_retrieved_date": retrieved,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output JSONL (default $DH_DATA_ROOT/raw/uspto.jsonl)")
    ap.add_argument("--cpc", default="A61K,A61P",
                    help="comma-separated CPC prefixes to keep (default A61K,A61P)")
    ap.add_argument("--since", type=int, default=None,
                    help="only patents with filing year >= this (e.g. 2010)")
    ap.add_argument("--max", type=int, default=0,
                    help="LIMIT rows (0 = all matching; use for an MVP slice)")
    ap.add_argument("--include-pregrant", action="store_true",
                    help="also include pre-grant publications (A1); default granted only")
    ap.add_argument("--project", default=None,
                    help="GCP billing project (default: GOOGLE_CLOUD_PROJECT / ADC)")
    ap.add_argument("--location", default="US", help="BigQuery job location")
    ap.add_argument("--dry-run", action="store_true",
                    help="estimate bytes scanned + cost, then exit without running")
    args = ap.parse_args()

    cpc_prefixes = [p.strip().upper() for p in args.cpc.split(",") if p.strip()]
    if not all(re.fullmatch(r"[A-Z0-9]+", p) for p in cpc_prefixes):
        print("ERROR: --cpc must be alphanumeric prefixes like A61K,A61P",
              file=sys.stderr)
        return 2

    try:
        from google.cloud import bigquery
    except ImportError:
        print("ERROR: google-cloud-bigquery not installed. Run:\n"
              "  pip install google-cloud-bigquery", file=sys.stderr)
        return 2

    try:
        client = bigquery.Client(project=args.project, location=args.location)
    except Exception as e:
        print(f"ERROR: could not create BigQuery client ({e}).\n"
              "  Set GOOGLE_APPLICATION_CREDENTIALS to a service-account key and "
              "GOOGLE_CLOUD_PROJECT to your project id.", file=sys.stderr)
        return 2

    sql = build_sql(cpc_prefixes, not args.include_pregrant, args.since,
                    args.max or None)

    # --- cost estimate (dry run) ---
    dry_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        est = client.query(sql, job_config=dry_cfg)
    except Exception as e:
        print(f"ERROR: query failed validation ({e}).\nSQL:\n{sql}", file=sys.stderr)
        return 1
    tib = est.total_bytes_processed / BYTES_PER_TIB
    billable_tib = max(0.0, tib - 1.0)      # first 1 TiB/month free
    print(f"Estimated scan: {est.total_bytes_processed / 1e9:.1f} GB "
          f"({tib:.3f} TiB). Cost if this is your first query this month: "
          f"${billable_tib * USD_PER_TIB:.2f} (first 1 TiB/month is free).")
    if args.dry_run:
        print("--dry-run: not executing. Re-run without --dry-run to fetch.")
        return 0

    # --- execute + stream to JSONL ---
    out_path = Path(args.out) if args.out else _data_root() / "raw" / "uspto.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved = date.today().isoformat()

    print(f"Running query -> {out_path} ...")
    job = client.query(sql)
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in job.result():                 # streams pages, low memory
            fh.write(json.dumps(row_to_record(row, retrieved),
                                ensure_ascii=False) + "\n")
            n += 1
            if n % 25000 == 0:
                print(f"  wrote {n:,} ...")
    scanned = (job.total_bytes_processed or 0) / 1e9
    print(f"\nDONE. {n:,} US pharma patents -> {out_path} "
          f"(scanned {scanned:.1f} GB). Stage 02 can parse it with the same "
          f"DH_DATA_ROOT set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
