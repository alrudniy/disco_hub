#!/usr/bin/env python3
"""
01_download_data.py  --  LAYER 1 (Evidence Integration), step 1 of 6.

Pulls raw records from each source into data/raw/<source>.jsonl.

  TARGET: Anvil CPU nodes for full scale (I/O + disk heavy); Drew is fine for the
          MVP slice (< ~200 GB).

Modes:
  --mock   generate a deterministic synthetic corpus (no network, no GPU). This
           is what CI and the smoke test use.
  (real)   hit the live APIs. ClinicalTrials.gov v2, OpenAlex, and SBIR are free
           and keyless; USPTO uses PatentsView; AUTM is supplied as pre-scraped
           JSONL by the students (no clean bulk API).

For full scale, do NOT page the OpenAlex REST API for 40M works -- download the
S3 snapshot instead:  aws s3 sync 's3://openalex' data/raw/openalex_snapshot \
    --no-sign-request   (~330 GB gzip). This script's REST path is for the MVP slice.

Usage:
  python 01_download_data.py --mock --mvp
  python 01_download_data.py --sources clinicaltrials,sbir --mvp
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

from discovery_hub import config
from discovery_hub.schema import write_jsonl
from discovery_hub import mock as mockgen


# Pharma-relevant CPC subclasses: A61K (medicinal preparations) and A61P
# (therapeutic activity). These are SUBCLASS codes -- the field below is
# cpc_current.cpc_subclass_id, not cpc_subgroup_id (subgroups look like
# "A61K31/00"). The old code filtered cpc_subgroup_id == "A61K", which matched
# nothing because A61K is a subclass value, not a subgroup value.
USPTO_PHARMA_SUBCLASSES = ("A61K", "A61P")


def _fetch_uspto(limit: int, api_key: str, subclasses=USPTO_PHARMA_SUBCLASSES,
                 since: str = "2010-01-01", sleep_s: float = 1.4) -> list[dict]:
    """
    Fetch pharma patents from the PatentsView PatentSearch API
    (https://search.patentsview.org/api/v1/patent/, CC-BY 4.0).

    Auth: the X-Api-Key header is REQUIRED (request a key from the PatentsView
    Help Center; set PATENTSVIEW_API_KEY). NOTE: the PatentSearch API is being
    migrated to the USPTO Open Data Portal (data.uspto.gov); expect interruptions
    and a future endpoint/key change -- see the ODP transition guide.

    Pagination uses the `after` cursor (sort by patent_id asc, size <= 1000). The
    request shape is GET with JSON-encoded q/f/s/o params (the documented
    preferred method). ~45 requests/min limit, so we sleep between pages.

    Output rows are normalized just enough that 02_parse_normalize.parse_uspto
    consumes them unchanged (cpc_codes + pharma_keywords populated from
    cpc_current; inventors/assignees left as nested objects).
    """
    import requests

    base = config.SOURCES["uspto"].base_url
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}
    q = {"_and": [
        {"_gte": {"patent_date": since}},
        {"_or": [{"cpc_current.cpc_subclass_id": sc} for sc in subclasses]},
    ]}
    f = ["patent_id", "patent_title", "patent_abstract", "patent_date",
         "inventors.inventor_name_first", "inventors.inventor_name_last",
         "assignees.assignee_organization",
         "cpc_current.cpc_subclass_id", "cpc_current.cpc_group_id"]
    s = [{"patent_id": "asc"}]

    out: list[dict] = []
    after = None
    while len(out) < limit:
        o = {"size": min(1000, limit - len(out))}
        if after:
            o["after"] = after
        params = {"q": json.dumps(q), "f": json.dumps(f),
                  "s": json.dumps(s), "o": json.dumps(o)}
        r = requests.get(base, params=params, headers=headers, timeout=60)
        if r.status_code == 429:  # throttled -- back off and retry
            time.sleep(5.0)
            continue
        if r.status_code in (401, 403):
            raise RuntimeError(
                "PatentSearch API rejected the key (HTTP %d). Check "
                "PATENTSVIEW_API_KEY, or note keys may need reissuing under the "
                "ODP migration." % r.status_code)
        r.raise_for_status()
        data = r.json()
        patents = data.get("patents") or []
        if not patents:
            break
        for p in patents:
            pid = p.get("patent_id", "")
            p["_source_url"] = f"https://patents.google.com/patent/{pid}"
            cpc = p.get("cpc_current") or []
            p["cpc_codes"] = [c.get("cpc_group_id") for c in cpc
                              if c.get("cpc_group_id")]
            p["pharma_keywords"] = sorted(
                {c.get("cpc_subclass_id") for c in cpc if c.get("cpc_subclass_id")})
            out.append(p)
        after = patents[-1].get("patent_id")
        if len(patents) < o["size"]:
            break
        time.sleep(sleep_s)
    return out[:limit]


# --------------------------------------------------------------------------- #
# USPTO Open Data Portal (ODP) -- the live replacement for the PatentSearch API.
# search.patentsview.org is being decommissioned in the ODP migration (its host
# currently does not resolve); ODP at api.uspto.gov is the forward path. ODP uses
# its own API keys (X-API-KEY header) obtained at data.uspto.gov "Getting Started"
# -- previously-issued PatentSearch keys do NOT work on ODP.
# --------------------------------------------------------------------------- #
ODP_API_BASE = "https://api.uspto.gov/api/v1"
# Patent File Wrapper "Entire Dataset": bibliographic / front-page patent data,
# JSON, 2001-present. The bulk replacement source for patent data on ODP.
ODP_DEFAULT_PRODUCT = "PTFWPRE"


def _odp_headers(api_key: str) -> dict:
    return {"X-API-KEY": api_key, "Accept": "application/json"}


def _fetch_uspto_odp_manifest(product_id: str, api_key: str) -> list[dict]:
    """
    List the downloadable bulk files for an ODP product via
    GET {ODP_API_BASE}/datasets/products/{product_id}. Endpoint and response shape
    (bulkDataProductBag[].productFileBag.fileDataBag[].fileDownloadURI) verified
    against the live ODP API.
    """
    import requests
    r = requests.get(f"{ODP_API_BASE}/datasets/products/{product_id}",
                     headers=_odp_headers(api_key), timeout=60)
    if r.status_code in (401, 403):
        raise RuntimeError(
            "ODP rejected the key (HTTP %d). Obtain an ODP API key at data.uspto.gov "
            "('Getting Started') and set USPTO_ODP_API_KEY -- old PatentSearch keys "
            "do not carry over." % r.status_code)
    r.raise_for_status()
    files = []
    for prod in r.json().get("bulkDataProductBag", []):
        for fb in (prod.get("productFileBag") or {}).get("fileDataBag", []):
            files.append({"fileName": fb.get("fileName"),
                          "uri": fb.get("fileDownloadURI"),
                          "from": fb.get("fileDataFromDate"),
                          "to": fb.get("fileDataToDate"),
                          "bytes": fb.get("fileSize")})
    return files


def _odp_download(uri: str, dest, api_key: str):
    """Stream a (large) ODP bulk file to disk."""
    import requests
    with requests.get(uri, headers=_odp_headers(api_key), stream=True,
                      timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def _fetch_sbir(limit: int, base_url: str, page_size: int = 100,
                sleep_s: float = 0.5, max_retries: int = 5) -> list[dict]:
    """
    Fetch SBIR/STTR awards from the SBIR.gov Awards API
    (https://api.www.sbir.gov/public/api/awards, public domain).

    The Awards API serves ~100 rows per request and pages with a `start` offset
    (start=0, 100, 200, ...); a single oversized request (e.g. rows=2000) is
    rejected with HTTP 429. So we page at `page_size` and walk `start`, backing
    off on 429 and sleeping politely between pages. Default sort is award-date desc.

    Each award is normalized just enough for 02_parse_normalize.parse_sbir:
    `_source_url` comes from award_link, and `award_id` is derived from it (the API
    returns no explicit id field) so doc_ids stay unique.

    NOTE: like the original, this pulls ALL agencies/topics unfiltered. For a
    pharma-relevant slice add e.g. params["research_area_keywords"]="..." (or filter
    downstream). For the *entire* corpus (~200k+ awards) the 290 MB bulk download is
    far faster than paging this rate-limited API.
    """
    import requests

    out: list[dict] = []
    start = 0
    while len(out) < limit:
        size = min(page_size, limit - len(out))
        params = {"rows": size, "start": start}
        retries = 0
        while True:
            r = requests.get(base_url, params=params, timeout=60)
            if r.status_code == 429:  # rate limited -- back off and retry this page
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(
                        "SBIR API kept returning HTTP 429 after %d retries at "
                        "start=%d. Lower page_size, raise sleep_s, or use the bulk "
                        "download (290 MB) instead of paging." % (max_retries, start))
                time.sleep(5.0)
                continue
            r.raise_for_status()
            break
        data = r.json()
        batch = data if isinstance(data, list) else (
            data.get("results") or data.get("data") or [])
        if not batch:
            break
        for rec in batch:
            link = rec.get("award_link") or ""
            rec["_source_url"] = link
            if not rec.get("award_id"):
                rec["award_id"] = (
                    link.rstrip("/").split("/")[-1] if link
                    else rec.get("agency_tracking_number")
                    or rec.get("contract") or f"row{len(out)}")
            out.append(rec)
        start += len(batch)
        if len(batch) < size:  # short page => no more results
            break
        time.sleep(sleep_s)
    return out[:limit]


def fetch_real(source: str, limit: int, uspto_backend: str = "patentsview",
               odp_download: bool = False) -> list[dict]:
    """Minimal real-API fetchers for the MVP slice. Robust paging is omitted for
    brevity; each returns up to `limit` raw records with a _source_url stamped."""
    import requests

    spec = config.SOURCES[source]
    out: list[dict] = []
    if source == "clinicaltrials":
        token = None
        while len(out) < limit:
            params = {"pageSize": min(1000, limit - len(out)),
                      "query.cond": "pharmaceutical OR cancer OR antibody"}
            if token:
                params["pageToken"] = token
            r = requests.get(spec.base_url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            for st in data.get("studies", []):
                nct = st.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                st["_source_url"] = f"https://clinicaltrials.gov/study/{nct}"
                out.append(st)
            token = data.get("nextPageToken")
            if not token:
                break
    elif source == "openalex":
        cursor = "*"
        concept = "C71924100"  # Medicine
        while len(out) < limit:
            params = {"filter": f"concepts.id:{concept}", "per-page": 200,
                      "cursor": cursor, "mailto": "tto-pipeline@example.org"}
            r = requests.get(spec.base_url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("results", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
    elif source == "sbir":
        out = _fetch_sbir(limit, spec.base_url)
    elif source == "uspto":
        if uspto_backend == "odp-bulk":
            api_key = os.environ.get("USPTO_ODP_API_KEY")
            if not api_key:
                print("  [uspto/odp-bulk] USPTO_ODP_API_KEY not set. Get an ODP key "
                      "at data.uspto.gov ('Getting Started') and export "
                      "USPTO_ODP_API_KEY. Skipping.", file=sys.stderr)
                return []
            try:
                files = _fetch_uspto_odp_manifest(ODP_DEFAULT_PRODUCT, api_key)
            except (RuntimeError, requests.exceptions.RequestException) as e:
                print(f"  [uspto/odp-bulk] {e}", file=sys.stderr)
                return []
            manifest = config.RAW_DIR / "uspto_odp_manifest.json"
            manifest.write_text(json.dumps(files, indent=2))
            gb = sum((f["bytes"] or 0) for f in files) / 1e9
            print(f"  [uspto/odp-bulk] {len(files)} bulk file(s), ~{gb:.1f} GB "
                  f"({ODP_DEFAULT_PRODUCT}). Manifest -> {manifest}", file=sys.stderr)
            if odp_download:
                for f in files:
                    dest = config.RAW_DIR / f["fileName"]
                    print(f"    downloading {f['fileName']} "
                          f"(~{(f['bytes'] or 0) / 1e9:.1f} GB) ...", file=sys.stderr)
                    _odp_download(f["uri"], dest, api_key)
            else:
                print("  [uspto/odp-bulk] manifest only (pass --odp-download to fetch "
                      "the zips). NOTE: bulk Patent File Wrapper JSON has a different "
                      "record shape than the PatentSearch API -- a dedicated parser is "
                      "needed before these feed 02_parse_normalize.", file=sys.stderr)
            return []  # bulk path produces files, not per-record dicts

        # default backend: PatentSearch API (currently in ODP migration).
        api_key = os.environ.get("PATENTSVIEW_API_KEY")
        if not api_key:
            print("  [uspto] PATENTSVIEW_API_KEY is not set. The PatentSearch API "
                  "requires a key (request one at the PatentsView Help Center, then "
                  "export PATENTSVIEW_API_KEY=...). Skipping USPTO.", file=sys.stderr)
            return []
        try:
            out = _fetch_uspto(limit, api_key)
        except requests.exceptions.ConnectionError:
            print("  [uspto] Could not reach search.patentsview.org -- the "
                  "PatentSearch API is being migrated to the USPTO Open Data Portal "
                  "and its host is currently unreachable. Re-run with "
                  "--uspto-backend odp-bulk (api.uspto.gov). See the ODP transition "
                  "guide.", file=sys.stderr)
            return []
    elif source == "autm":
        print(f"  [autm] no bulk API -- expects students to drop pre-scraped JSONL "
              f"at {config.RAW_DIR / 'autm.jsonl'}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="generate synthetic data")
    ap.add_argument("--mvp", action="store_true", help="use small MVP volumes")
    ap.add_argument("--sources", default=",".join(config.SOURCES),
                    help="comma-separated subset of sources")
    ap.add_argument("--limit", type=int, default=None,
                    help="override per-source record count")
    ap.add_argument("--uspto-backend", choices=["patentsview", "odp-bulk"],
                    default="patentsview",
                    help="USPTO source: patentsview (PatentSearch API, in ODP "
                         "migration) or odp-bulk (data.uspto.gov bulk datasets)")
    ap.add_argument("--odp-download", action="store_true",
                    help="with --uspto-backend odp-bulk, download the bulk zips "
                         "(multi-GB) instead of just writing the file manifest")
    args = ap.parse_args()

    config.ensure_dirs()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    total = 0
    for src in sources:
        spec = config.SOURCES[src]
        limit = args.limit or (spec.mvp_records if args.mvp else spec.mvp_records * 10)
        if args.mock:
            records = list(mockgen.GENERATORS[src](limit, config.SEED))
        else:
            endpoint = spec.base_url or "local"
            if src == "uspto" and args.uspto_backend == "odp-bulk":
                endpoint = f"{ODP_API_BASE}/datasets/products/{ODP_DEFAULT_PRODUCT}"
            print(f"[{src}] fetching up to {limit} from {endpoint} "
                  f"(license: {spec.license})")
            records = fetch_real(src, limit, args.uspto_backend, args.odp_download)
        # Stamp retrieved_date at the raw layer too (provenance convention).
        for rec in records:
            rec.setdefault("_retrieved_date", date.today().isoformat())
        path = config.RAW_DIR / f"{src}.jsonl"
        n = write_jsonl(records, path)
        total += n
        print(f"[{src}] wrote {n:>6} records -> {path}")
    print(f"\nDONE. {total} raw records across {len(sources)} sources in {config.RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
