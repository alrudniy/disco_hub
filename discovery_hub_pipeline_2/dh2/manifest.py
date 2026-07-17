"""
dh2.manifest -- write the version metadata every model/index artifact must carry (S9.2).

Also computes the document-order checksum that guards the index-alignment invariant:
a hash of the doc_ids order. If two artifacts share this checksum, their row orders
match and their vectors/index are interchangeable; if not, they must not be mixed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def doc_order_checksum(doc_ids: list[str]) -> str:
    """Stable hash of doc_id ORDER (the index-alignment fingerprint)."""
    h = hashlib.sha256()
    for d in doc_ids:
        h.update(d.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _git_commit(repo_dir: str | None) -> str:
    if not repo_dir:
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_manifest(path: str | Path, *, artifact_kind: str,
                   base_model: str = "", train_data_version: str = "",
                   query_generator: str = "", teacher_models: list[str] | None = None,
                   objective: str = "", hyperparams: dict[str, Any] | None = None,
                   query_prefix: str = "", document_prefix: str = "",
                   max_seq_length: int = 512, embedding_dim: int = 0,
                   doc_ids: list[str] | None = None, qrels_version: str = "",
                   eval_report_path: str = "", repo_dir: str | None = None,
                   extra: dict[str, Any] | None = None) -> str:
    """Write a §9.2-compliant manifest JSON next to an artifact. Returns the path."""
    manifest = {
        "artifact_kind": artifact_kind,
        "source_commit": _git_commit(repo_dir),
        # base_model was accepted by this signature and silently dropped from the written
        # manifest, so no artifact on disk recorded which model produced its vectors. That
        # is how shards 0-2 of the 4B matrix kept a different model's vectors (cos ~= 0.00
        # vs a re-encode) through a resumed run, undetected, for 40.7% of clinical trials.
        "base_model": base_model,
        "train_data_version": train_data_version,
        "query_generator_version": query_generator,
        "teacher_model_versions": teacher_models or [],
        "objective": objective,
        "hyperparameters": hyperparams or {},
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
        "max_sequence_length": max_seq_length,
        "embedding_dim": embedding_dim,
        "document_order_checksum": doc_order_checksum(doc_ids) if doc_ids else "",
        "qrels_version": qrels_version,
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluation_report_path": eval_report_path,
    }
    if extra:
        manifest["extra"] = extra
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2))
    return str(p)


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def compatible(m1: dict, m2: dict) -> bool:
    """Two artifacts are row-compatible iff same model, same encode settings, same dim,
    and same doc-order checksum.

    dim + doc-order alone is NOT sufficient and never was. Both are invariant to WHICH
    MODEL wrote the vectors: two matrices embedded by different checkpoints over the same
    corpus in the same order have identical checksums and identical dims, and this
    function called them interchangeable. That is precisely the dark-shard failure --
    same order, same dim, orthogonal vectors. base_model and max_sequence_length are the
    fields that actually discriminate, so they are required here.
    """
    if not m1.get("document_order_checksum"):
        return False
    for field in ("embedding_dim", "document_order_checksum",
                  "base_model", "max_sequence_length"):
        if m1.get(field) != m2.get(field):
            return False
    # A manifest predating the base_model fix records "" and cannot prove provenance.
    return bool(m1.get("base_model"))
