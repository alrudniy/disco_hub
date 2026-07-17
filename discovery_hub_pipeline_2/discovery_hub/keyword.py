"""
BM25 keyword retrieval -- the lexical half of hybrid search.

The dense encoder is good at semantics but blind to exact strings: gene symbols,
acronyms (GLP-1, PD-L1, CAR-T), assay names, chemical identifiers. BM25 is the
complement -- it rewards rare exact-term matches. Fusing the two (see fusion.py)
is what makes the deck's "hybrid semantic search" claim literally true instead of
aspirational; right now the pipeline only had the dense half.

This is a faithful BM25-Okapi implementation over an inverted index (postings),
so scoring a query only touches documents that contain a query term -- the same
structure Lucene/OpenSearch use. It is pure Python, deterministic, and
unit-tested (tests/test_hybrid.py). `rank_bm25.BM25Okapi` is a drop-in
alternative with the identical formula; this in-package version avoids a runtime
dependency and gives byte-stable persistence for the reproducibility story.

At production scale (millions of docs) the persisted JSON postings should become
a real search service (OpenSearch / Tantivy); the Retriever interface does not
change when you make that swap.
"""
from __future__ import annotations

import json
import math
import re

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, length >= 2 (keeps acronyms like 'il6')."""
    return [t for t in _TOK.findall((text or "").lower()) if len(t) >= 2]


class BM25Index:
    """Inverted-index BM25-Okapi over a fixed corpus."""

    def __init__(self, doc_ids, doc_len, avgdl, idf, postings,
                 k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids                    # row -> doc_id
        self.doc_len = doc_len                    # row -> token count
        self.avgdl = avgdl
        self.idf = idf                            # term -> idf
        self.postings = postings                  # term -> [[row, tf], ...]
        self.k1 = k1
        self.b = b

    # -- construction --------------------------------------------------------
    @classmethod
    def build(cls, docs, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        """
        Build from DiscoveryDoc-like objects (need .doc_id and .embedding_text;
        falls back to title+abstract). Deterministic: depends only on corpus.
        """
        doc_ids: list[str] = []
        doc_len: list[int] = []
        postings: dict[str, list[list[int]]] = {}
        df: dict[str, int] = {}

        for row, d in enumerate(docs):
            text = getattr(d, "embedding_text", "") or \
                f"{getattr(d, 'title', '')} {getattr(d, 'abstract', '')}"
            toks = tokenize(text)
            doc_ids.append(d.doc_id)
            doc_len.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, c in tf.items():
                postings.setdefault(t, []).append([row, c])
                df[t] = df.get(t, 0) + 1

        n = len(doc_ids)
        avgdl = (sum(doc_len) / n) if n else 0.0
        # BM25-Okapi idf with the +1 smoothing that keeps idf >= 0.
        idf = {t: math.log(1 + (n - dft + 0.5) / (dft + 0.5)) for t, dft in df.items()}
        return cls(doc_ids, doc_len, avgdl, idf, postings, k1=k1, b=b)

    # -- scoring -------------------------------------------------------------
    def score(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Return up to top_k (doc_id, bm25_score) for the query, best first. Only
        documents containing a query term are scored. Ties broken by doc_id for
        deterministic ordering (the reproducibility story depends on this).
        """
        q_terms = set(tokenize(query))
        scores: dict[int, float] = {}
        for t in q_terms:
            post = self.postings.get(t)
            if not post:
                continue
            idf = self.idf.get(t, 0.0)
            for row, tf in post:
                dl = self.doc_len[row]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[row] = scores.get(row, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        return [(self.doc_ids[row], s) for row, s in ranked[:top_k]]

    # -- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "doc_ids": self.doc_ids, "doc_len": self.doc_len, "avgdl": self.avgdl,
            "idf": self.idf, "postings": self.postings, "k1": self.k1, "b": self.b,
        }))

    @classmethod
    def load(cls, path) -> "BM25Index":
        o = json.loads(path.read_text())
        return cls(o["doc_ids"], o["doc_len"], o["avgdl"], o["idf"],
                   o["postings"], k1=o.get("k1", 1.5), b=o.get("b", 0.75))
