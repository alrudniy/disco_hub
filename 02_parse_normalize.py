#!/usr/bin/env python3
"""
02_parse_normalize.py  --  LAYER 1, step 2 of 6.

Maps every heterogeneous raw schema in data/raw/*.jsonl into the unified
DiscoveryDoc and writes data/normalized/docs.jsonl. Each source gets one parser;
all of them converge on the same record type with a canonical embedding_text.

  TARGET: Anvil CPU (or Drew for the MVP). Pure CPU, no GPU.

This is the extension of discovery_finetune's parse.py to the fuller source set
(adds OpenAlex + SBIR on top of AUTM / ClinicalTrials.gov / USPTO).

Usage:
  python 02_parse_normalize.py
"""
from __future__ import annotations

import argparse

from discovery_hub import config
from discovery_hub.schema import DiscoveryDoc, read_jsonl, write_jsonl


def _openalex_abstract(inv_index: dict | None) -> str:
    """Reconstruct plain text from OpenAlex's abstract_inverted_index."""
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def parse_clinicaltrials(rec: dict) -> DiscoveryDoc:
    ps = rec.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    desc = ps.get("descriptionModule", {})
    spon = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    locs = ps.get("contactsLocationsModule", {}).get("locations", []) or []
    nct = ident.get("nctId", "")
    doc = DiscoveryDoc(
        doc_id=f"clinicaltrials:{nct}",
        source="clinicaltrials",
        title=ident.get("briefTitle", ""),
        abstract=desc.get("briefSummary", ""),
        organizations=[spon["name"]] if spon.get("name") else [],
        facilities=[l.get("facility", "") for l in locs if l.get("facility")],
        source_url=rec.get("_source_url", f"https://clinicaltrials.gov/study/{nct}"),
        retrieved_date=rec.get("_retrieved_date", ""),
    )
    return doc


def parse_uspto(rec: dict) -> DiscoveryDoc:
    def _names(items, key):
        out = []
        for it in items or []:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                out.append(it.get(key) or it.get("name") or
                           f"{it.get('inventor_name_first','')} {it.get('inventor_name_last','')}".strip())
        return [x for x in out if x]

    pid = rec.get("patent_id", "")
    return DiscoveryDoc(
        doc_id=f"uspto:{pid}",
        source="uspto",
        title=rec.get("patent_title", ""),
        abstract=rec.get("patent_abstract", ""),
        inventors=_names(rec.get("inventors"), "inventor_name_last"),
        organizations=_names(rec.get("assignees"), "assignee_organization"),
        cpc_codes=rec.get("cpc_codes", []),
        keywords=rec.get("pharma_keywords", []),
        source_url=rec.get("_source_url", f"https://patents.google.com/patent/{pid}"),
        retrieved_date=rec.get("_retrieved_date", ""),
    )


def parse_openalex(rec: dict) -> DiscoveryDoc:
    wid = rec.get("id", "").rsplit("/", 1)[-1]
    authorships = rec.get("authorships", []) or []
    experts = [a.get("author", {}).get("display_name", "") for a in authorships]
    orgs = []
    for a in authorships:
        for inst in a.get("institutions", []) or []:
            if inst.get("display_name"):
                orgs.append(inst["display_name"])
    concepts = [c.get("display_name", "") for c in rec.get("concepts", []) or []]
    return DiscoveryDoc(
        doc_id=f"openalex:{wid}",
        source="openalex",
        title=rec.get("title") or "",
        abstract=_openalex_abstract(rec.get("abstract_inverted_index")),
        experts=[e for e in experts if e],
        organizations=list(dict.fromkeys(orgs)),
        keywords=[c for c in concepts if c],
        source_url=rec.get("id", ""),
        retrieved_date=rec.get("_retrieved_date", ""),
    )


def parse_sbir(rec: dict) -> DiscoveryDoc:
    aid = rec.get("award_id", "")
    return DiscoveryDoc(
        doc_id=f"sbir:{aid}",
        source="sbir",
        title=rec.get("award_title", ""),
        abstract=rec.get("abstract", ""),
        organizations=[rec["firm"]] if rec.get("firm") else [],
        experts=[rec["pi_name"]] if rec.get("pi_name") else [],
        source_url=rec.get("_source_url", ""),
        retrieved_date=rec.get("_retrieved_date", ""),
    )


def parse_autm(rec: dict) -> DiscoveryDoc:
    lid = rec.get("listing_id", "")
    return DiscoveryDoc(
        doc_id=f"autm:{lid}",
        source="autm",
        title=rec.get("tech_title", ""),
        abstract=rec.get("description", ""),
        inventors=rec.get("inventors", []),
        organizations=[rec["university"]] if rec.get("university") else [],
        source_url=rec.get("_source_url", ""),
        retrieved_date=rec.get("_retrieved_date", ""),
    )


PARSERS = {
    "clinicaltrials": parse_clinicaltrials,
    "uspto": parse_uspto,
    "openalex": parse_openalex,
    "sbir": parse_sbir,
    "autm": parse_autm,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-chars", type=int, default=20,
                    help="drop docs whose abstract is shorter than this (AUTM noise filter)")
    args = ap.parse_args()

    config.ensure_dirs()
    docs: list[DiscoveryDoc] = []
    dropped = 0
    for src, parser in PARSERS.items():
        path = config.RAW_DIR / f"{src}.jsonl"
        if not path.exists():
            continue
        for raw in read_jsonl(path):
            doc = parser(raw)
            if len((doc.abstract or "").strip()) < args.min_chars:
                dropped += 1
                continue
            doc.build_embedding_text()
            docs.append(doc)

    out = config.NORM_DIR / "docs.jsonl"
    n = write_jsonl(docs, out)
    by_src: dict[str, int] = {}
    for d in docs:
        by_src[d.source] = by_src.get(d.source, 0) + 1
    print(f"Normalized {n} DiscoveryDocs (dropped {dropped} short/noise) -> {out}")
    for s, c in sorted(by_src.items()):
        print(f"  {s:>16}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
