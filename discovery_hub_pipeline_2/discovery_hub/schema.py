"""
The unified data contract for the whole pipeline.

`DiscoveryDoc` is the single normalized record type that every heterogeneous
source (AUTM pages, ClinicalTrials.gov v2, USPTO patents, OpenAlex works, SBIR
awards) is mapped into by 02_parse_normalize.py. Downstream stages only ever see
DiscoveryDoc, never the raw source schemas -- this is what makes the students'
data-collection work parallelizable behind one contract.

Every record carries `source_url` and `retrieved_date`, the standing convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

# --------------------------------------------------------------------------- #
# Graph vocabulary (Layer 1). Node and relation type strings are frozen here so
# 03_build_graph.py and 06_train_rgcn.py agree on the R-GCN relation index.
# --------------------------------------------------------------------------- #
NODE_TYPES = ("technology", "inventor", "organization", "expert", "facility")

RELATIONS = (
    "invented_by",     # technology -> inventor
    "assigned_to",     # technology -> organization
    "investigated_by", # technology -> expert (PI / author)
    "located_at",      # technology -> facility
    "affiliated_with", # inventor/expert -> organization
)

NODE_TYPE_TO_ID = {t: i for i, t in enumerate(NODE_TYPES)}
RELATION_TO_ID = {r: i for i, r in enumerate(RELATIONS)}


# --------------------------------------------------------------------------- #
# The unified document
# --------------------------------------------------------------------------- #
@dataclass
class DiscoveryDoc:
    doc_id: str                 # globally unique, e.g. "uspto:US10123456B2"
    source: str                 # one of SOURCES keys
    title: str
    abstract: str
    # Typed entities -> become graph nodes/edges in Layer 1.
    inventors: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    experts: list[str] = field(default_factory=list)        # PIs, authors
    facilities: list[str] = field(default_factory=list)
    cpc_codes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # Provenance (standing convention).
    source_url: str = ""
    retrieved_date: str = field(default_factory=lambda: date.today().isoformat())
    # The canonical text that gets embedded. Filled by build_embedding_text().
    embedding_text: str = ""
    extra: dict = field(default_factory=dict)

    def build_embedding_text(self) -> str:
        """
        Construct the canonical embedding text. We front-load the title and
        abstract (highest signal) and append a compact entity/keyword tail so the
        encoder sees who/where as well as what. Keeping this deterministic and in
        one place is important for reproducibility: the same DiscoveryDoc always
        yields the same embedding_text, hence the same vector.
        """
        parts = [self.title.strip(), self.abstract.strip()]
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords[:12]))
        if self.organizations:
            parts.append("Organizations: " + ", ".join(self.organizations[:6]))
        if self.cpc_codes:
            parts.append("CPC: " + ", ".join(self.cpc_codes[:8]))
        text = "\n".join(p for p in parts if p)
        self.embedding_text = text
        return text


# --------------------------------------------------------------------------- #
# JSONL I/O -- the storage format for every inter-stage artifact.
# --------------------------------------------------------------------------- #
def write_jsonl(records: Iterable, path: Path) -> int:
    """Write dataclass instances or dicts to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            obj = asdict(rec) if hasattr(rec, "__dataclass_fields__") else rec
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream dicts from a JSONL file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_docs(path: Path) -> Iterator[DiscoveryDoc]:
    """Stream DiscoveryDoc objects from a normalized JSONL file."""
    for obj in read_jsonl(path):
        yield DiscoveryDoc(**obj)
