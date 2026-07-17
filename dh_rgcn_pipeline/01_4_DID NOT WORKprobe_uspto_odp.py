#!/usr/bin/env python3
"""
probe_uspto_odp.py  --  discover the ODP Patent File Wrapper search schema.

Before writing the real fetcher we need to know TWO things the USPTO docs don't
make clear: (1) does the PFW search response include the patent ABSTRACT (essential
for embedding), and (2) what is the CPC classification field actually called (so we
can filter to pharma A61K/A61P). This fetches a couple of records with your key and
prints the real field layout so those can be pinned down.

Run on Drew (needs your ODP key, no third-party deps):
    export USPTO_ODP_API_KEY='mwkpvbljidnnndutrlxajwxknbscdr'
    python probe_uspto_odp.py
Then paste the three labeled sections back.
"""
import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://api.uspto.gov/api/v1/patent/applications/search"


def post(body: dict, key: str) -> dict:
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"X-API-KEY": key, "Content-Type": "application/json",
                 "Accept": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def walk(obj, prefix=""):
    """Yield (path, leaf_value) for the whole nested record; lists shown via [0]."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], (dict, list)):
            yield from walk(obj[0], prefix + "[0]")
        else:
            yield (prefix, obj)
    else:
        yield (prefix, obj)


def main() -> int:
    key = os.environ.get("USPTO_ODP_API_KEY")
    if not key:
        print("Set USPTO_ODP_API_KEY first (export USPTO_ODP_API_KEY=...).",
              file=sys.stderr)
        return 2

    # Correct ODP request shape: `pagination` is a NESTED object -- a top-level
    # limit/offset is what caused the 400. Try a few forms so we get through; `q`
    # is optional free-form full-text search ("antibody" -> pharma-relevant records
    # so CPC + abstract fields, if present, show up populated).
    candidates = [
        {"q": "antibody", "pagination": {"offset": 0, "limit": 3}},
        {"pagination": {"offset": 0, "limit": 3}},   # blank q, just paginate
        {"q": "antibody"},                            # no pagination object
    ]
    data = None
    for i, body in enumerate(candidates, 1):
        try:
            data = post(body, key)
            print(f"(request form {i} accepted: {json.dumps(body)})")
            break
        except urllib.error.HTTPError as e:
            print(f"form {i} -> HTTP {e.code}: {e.read().decode()[:400]}",
                  file=sys.stderr)
        except Exception as e:
            print(f"form {i} -> ERROR: {e}", file=sys.stderr)
    if data is None:
        print("\nAll request forms failed -- paste the errors above and I'll adjust.",
              file=sys.stderr)
        return 1

    total = data.get("totalNumFound")
    bag = data.get("patentFileWrapperDataBag") or []
    print(f"totalNumFound = {total}   returned = {len(bag)}")
    if not bag:
        print("No records returned. Full response envelope:")
        print(json.dumps(data, indent=2)[:2000])
        return 0

    paths = list(walk(bag[0]))

    print("\n=== [1] ALL field paths in the first record ===")
    for p, v in paths:
        s = str(v)
        print(f"  {p} = {s[:90] + '...' if len(s) > 90 else s}")

    print("\n=== [2] fields matching /abstract/i ===")
    ah = [(p, v) for p, v in paths if "abstract" in p.lower()]
    if ah:
        for p, v in ah:
            print(f"  {p} = {str(v)[:200]}")
    else:
        print("  NONE  <-- PFW search does NOT return an abstract; we'll need a "
              "different source for abstract text.")

    print("\n=== [3] fields matching /cpc|classification/i ===")
    ch = [(p, v) for p, v in paths if "cpc" in p.lower() or "classif" in p.lower()]
    if ch:
        for p, v in ch:
            print(f"  {p} = {str(v)[:200]}")
    else:
        print("  NONE found under those names -- paste section [1] so I can spot it.")

    print("\nPaste sections [1]-[3] back and I'll finalize the fetcher's field map "
          "(and the exact CPC filter query).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
