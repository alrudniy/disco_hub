"""Build the gene/protein alias -> canonical-symbol table from HGNC.

ONE artifact, THREE uses -- and two of them pull in OPPOSITE directions:

  1. SERVING query expansion. A scout typing PD-L1 should reach a patent that only ever
     says CD274. Expand the query.
  2. §5 EXAMPLE CHECKING -- NORMALIZED. An alias swap is still a restatement of the
     document. If a hand-written example says "PD-L1" where the document says "B7-H1",
     the raw overlap metric cannot see the match and scores it as a good low-overlap
     example. It is not: it is a paraphrase wearing a different name. Normalize first,
     then measure, or the §5 gate rewards alias-swapping.
  3. EVAL BUCKETING -- RAW, deliberately NOT normalized. A PD-L1/B7-H1 pair genuinely IS
     hard for a lexical matcher, and that difficulty is the thing the register split is
     supposed to capture. Normalizing here would erase the phenomenon being measured.

Same table, opposite direction. Use `normalize=` explicitly at every call site.

Source: HGNC complete set (public domain).
  https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Aliases too generic/ambiguous to normalize on: real English words, or symbols that
# collide with common tokens. Mapping these would create false matches -- worse than the
# misses we are fixing.
_UNSAFE = {
    "T", "A", "B", "C", "D", "E", "S", "P", "N", "G", "H", "M", "L", "R", "X", "Y",
    "CAT", "CAR", "SET", "MET", "REST", "WAS", "CAN", "MAY", "END", "ACHE", "AGE",
    "AIM", "ARM", "ART", "BAD", "BAG", "BAT", "BID", "BIN", "CAP", "CASE", "CELL",
    "COIL", "COPE", "CS", "DAD", "DAP", "DDT", "ECM", "EMS", "ETS", "FACE", "FAM",
    "FAN", "FAT", "GAS", "GEM", "HAND", "HAT", "HEY", "HR", "ICE", "IMPACT", "IPO",
    "ITCH", "JAW", "KID", "LAG", "LAP", "LIP", "MAD", "MAP", "MAX", "MICE", "MIC",
    "MIR", "MOB", "NET", "NICE", "NIP", "NOT", "PAN", "PAR", "PET", "PIG", "PIN",
    "PLAN", "POP", "RAG", "RAIN", "RAN", "SAG", "SALL", "SAP", "SDS", "SHE", "SIP",
    "SON", "SOS", "SPIN", "SUN", "TAF", "TAP", "TEC", "TIP", "TOP", "TRAM", "TRIP",
    "VIP", "WARS", "WAVE", "WEE", "WIT",
}


def _norm_key(s: str) -> str:
    """Match dh2.pharma_tok's canonical identifier form: lowercase, hyphens collapsed."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def build(hgnc_tsv: str | Path, out_path: str | Path) -> dict:
    alias_to_symbol: dict[str, str] = {}
    claims: dict[str, set[str]] = defaultdict(set)
    n_genes = 0

    with open(hgnc_tsv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("status") != "Approved":
                continue
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            n_genes += 1
            names = [sym]
            for col in ("alias_symbol", "prev_symbol"):
                raw = (row.get(col) or "").strip()
                if raw:
                    names += [a.strip() for a in raw.split("|") if a.strip()]
            for a in names:
                if a.upper() in _UNSAFE or len(a) < 2:
                    continue
                k = _norm_key(a)
                if len(k) < 3:            # "pd", "b7" alone are not identifiers
                    continue
                claims[k].add(sym)

    # An alias claimed by two different genes is ambiguous -- dropping it is the safe
    # default. Normalizing IL1 to the wrong gene is worse than not normalizing it.
    ambiguous = {k for k, v in claims.items() if len(v) > 1}
    for k, v in claims.items():
        if k not in ambiguous:
            alias_to_symbol[k] = next(iter(v))

    out = {
        "_source": "HGNC complete set (Approved only)",
        "_n_genes": n_genes,
        "_n_aliases": len(alias_to_symbol),
        "_n_dropped_ambiguous": len(ambiguous),
        "alias_to_symbol": alias_to_symbol,
    }
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/hgnc_complete_set.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/workspace/dh_data_v2/gene_synonyms.json"
    o = build(src, dst)
    print(f"genes (Approved)      : {o['_n_genes']}")
    print(f"aliases mapped        : {o['_n_aliases']}")
    print(f"dropped as ambiguous  : {o['_n_dropped_ambiguous']}")
    print(f"wrote {dst}")
    m = o["alias_to_symbol"]
    print("\n=== the case that started this ===")
    for a in ("PD-L1", "PDL1", "B7-H1", "B7H1", "CD274", "PD-1", "PDCD1", "HER2", "ERBB2"):
        print(f"  {a:8s} -> {m.get(_norm_key(a), '(unmapped)')}")
