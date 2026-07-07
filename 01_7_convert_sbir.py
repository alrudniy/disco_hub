#!/usr/bin/env python3
"""
RUN THIS LINE FIRST
export DH_DATA_ROOT=/home/alex/discovery_hub/data
THEN RUN 
python3 01_7_convert_sbir.py --require-abstract

01_6_convert_sbir.py  --  SBIR bulk CSV -> pharma-filtered, parse-ready JSONL.

The SBIR "download all awards" file is CSV (despite a .json name) and covers every
agency -- the vast majority are DoD/DHS/NASA, not biomedical. SBIR records carry no
CPC or any clean subject field, so relevance has to come from the funding agency
plus a therapeutics keyword match. This remaps the CSV columns into the shape
02_parse_normalize.parse_sbir reads and keeps only bio/pharma awards.

Column map (CSV -> parse_sbir):
  Company -> firm | Award Title -> award_title | Abstract -> abstract
  PI Name -> pi_name | Contract (or Agency Tracking Number) -> award_id
There is no per-award URL column, so _source_url is left empty.

Filter modes (--mode):
  agency-or-keyword  (default) keep if HHS/NIH-funded OR a therapeutics keyword hits
  agency-only        highest precision: only Health & Human Services (NIH) awards
  keyword-only       only the keyword match, any agency
  all                no filter (remap everything ~200k)

Examples:
  python 01_6_convert_sbir.py                         # default filter -> raw/sbir.jsonl
  python 01_6_convert_sbir.py --mode agency-only --require-abstract
  python 01_6_convert_sbir.py --in raw/sbir_bulk.json --max 20000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

# SBIR agencies that are essentially all biomedical (NIH lives under HHS).
DEFAULT_AGENCIES = ["Health and Human Services", "National Institutes of Health"]

# Therapeutics / drug-development lexicon (case-insensitive substring match).
DEFAULT_KEYWORDS = [
    "therap", "drug", "pharmaceutic", "vaccine", "antibod", "antiviral",
    "antibiotic", "antimicrob", "oncolog", "cancer", "tumor", "carcinoma",
    "leukemia", "immunother", "chemother", "biomarker", "small molecule",
    "inhibitor", "peptide", "gene therapy", "crispr", "monoclonal", "clinical",
    "preclinical", "in vivo", "biologic", "mrna", "sirna", "protease", "kinase",
    "pathogen", "infectious disease", "therapeutic target",
]

# CSV column names (from the bulk file header).
C_FIRM, C_TITLE, C_ABSTRACT, C_PI = "Company", "Award Title", "Abstract", "PI Name"
C_AGENCY, C_CONTRACT, C_TRACKING = "Agency", "Contract", "Agency Tracking Number"
C_YEAR = "Award Year"


def _data_root() -> Path:
    return Path(os.environ.get("DH_DATA_ROOT", "./data")).resolve()


def make_award_id(row: dict) -> str:
    aid = (row.get(C_CONTRACT) or "").strip() or (row.get(C_TRACKING) or "").strip()
    if not aid:
        seed = (row.get(C_FIRM, "") + row.get(C_TITLE, "") + row.get(C_YEAR, ""))
        aid = "h" + hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return re.sub(r"\s+", "_", aid)


def in_scope(row: dict, agencies: list[str], keyword_re, mode: str) -> bool:
    if mode == "all":
        return True
    agency = row.get(C_AGENCY, "") or ""
    agency_hit = any(a.lower() in agency.lower() for a in agencies)
    if mode == "agency-only":
        return agency_hit
    text = f"{row.get(C_TITLE, '')} {row.get(C_ABSTRACT, '')}".lower()
    kw_hit = bool(keyword_re.search(text))
    if mode == "keyword-only":
        return kw_hit
    return agency_hit or kw_hit          # agency-or-keyword (default)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=None,
                    help="SBIR bulk CSV (default $DH_DATA_ROOT/raw/sbir_bulk.json)")
    ap.add_argument("--out", default=None,
                    help="output JSONL (default $DH_DATA_ROOT/raw/sbir.jsonl)")
    ap.add_argument("--mode", default="agency-or-keyword",
                    choices=["agency-or-keyword", "agency-only", "keyword-only", "all"])
    ap.add_argument("--keywords", default=None,
                    help="comma-separated override of the therapeutics lexicon")
    ap.add_argument("--require-abstract", action="store_true",
                    help="drop awards with an empty abstract")
    ap.add_argument("--max", type=int, default=0, help="cap kept records (0 = all)")
    args = ap.parse_args()

    root = _data_root()
    in_path = Path(args.in_path) if args.in_path else root / "raw" / "sbir_bulk.json"
    out_path = Path(args.out) if args.out else root / "raw" / "sbir.jsonl"
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    keywords = ([k.strip() for k in args.keywords.split(",") if k.strip()]
                if args.keywords else DEFAULT_KEYWORDS)
    keyword_re = re.compile("|".join(re.escape(k) for k in keywords), re.I)

    # abstracts can be long; lift the CSV field cap.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieved = date.today().isoformat()
    total = kept = no_abstract = 0

    # utf-8-sig strips a BOM if present; newline="" lets csv handle embedded newlines.
    with open(in_path, "r", encoding="utf-8-sig", newline="") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            total += 1
            if not in_scope(row, DEFAULT_AGENCIES, keyword_re, args.mode):
                continue
            abstract = (row.get(C_ABSTRACT) or "").strip()
            if args.require_abstract and not abstract:
                no_abstract += 1
                continue
            rec = {
                "award_id": make_award_id(row),
                "award_title": (row.get(C_TITLE) or "").strip(),
                "abstract": abstract,
                "firm": (row.get(C_FIRM) or "").strip(),
                "pi_name": (row.get(C_PI) or "").strip(),
                "_source_url": "",          # no per-award link in the bulk CSV
                "_retrieved_date": retrieved,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if args.max and kept >= args.max:
                break

    print(f"SBIR convert [{args.mode}]: read {total:,} awards, kept {kept:,}"
          + (f" (skipped {no_abstract:,} with no abstract)" if args.require_abstract else ""))
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
