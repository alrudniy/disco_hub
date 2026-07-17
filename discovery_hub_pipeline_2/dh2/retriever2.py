"""
dh2.retriever2 -- a minimal, self-contained dense retriever for candidate pooling.

Unlike the production stage-07 Retriever (which is wired to the deployed model/index),
this loads an ARBITRARY (model_dir, doc_vectors.npy, doc_ids.json) triple, so the
candidate-pool builder can pull from the 4B AND 8B models (and deeper rank windows)
in the same process. Vectors are assumed L2-normalized (IndexFlatIP == cosine).

Kept dependency-light: numpy exact search by default (identical top-k to FAISS Flat),
optional FAISS if installed and requested.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from discovery_hub.embedding import format_query
from discovery_hub import config as base


class DenseRetriever:
    def __init__(self, model_dir: str, doc_vectors_path: str, doc_ids_path: str,
                 device: str | None = None, query_instruction: str | None = None,
                 use_faiss: bool = False, batch_size: int = 64):
        self.model_dir = model_dir
        self.device = device
        self.query_instruction = query_instruction or base.QUERY_INSTRUCTION
        self.batch_size = batch_size
        self.doc_vectors = np.load(doc_vectors_path, mmap_mode="r")
        self.doc_ids = json.loads(Path(doc_ids_path).read_text())
        assert len(self.doc_ids) == self.doc_vectors.shape[0], (
            f"doc_ids ({len(self.doc_ids)}) != vectors ({self.doc_vectors.shape[0]})")
        self.dim = int(self.doc_vectors.shape[1])
        self._model = None
        self._faiss = None
        if use_faiss:
            try:
                import faiss
                idx = faiss.IndexFlatIP(self.dim)
                idx.add(np.ascontiguousarray(self.doc_vectors, dtype=np.float32))
                self._faiss = idx
            except Exception:
                self._faiss = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_dir, device=self.device)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a scout query with the Qwen3 instruction prefix (query side)."""
        m = self._load_model()
        # format_query prepends "Instruct: {task}\nQuery: "; SentenceTransformer encodes it.
        text = format_query(query, self.query_instruction)
        v = m.encode([text], batch_size=1, normalize_embeddings=True,
                     show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)[0]

    def search(self, query: str, top_k: int = 100,
               window: tuple[int, int] | None = None) -> list[tuple[str, float]]:
        """Return [(doc_id, cosine)] for the top_k (or a [start,end) rank window)."""
        qv = self.encode_query(query)
        depth = top_k if window is None else max(window)
        if self._faiss is not None:
            sc, idx = self._faiss.search(qv.reshape(1, -1), depth)
            pairs = [(self.doc_ids[i], float(sc[0][r])) for r, i in enumerate(idx[0])]
        else:
            sims = np.asarray(self.doc_vectors, dtype=np.float32) @ qv
            top = np.argpartition(-sims, min(depth, len(sims) - 1))[:depth]
            top = top[np.argsort(-sims[top])]
            pairs = [(self.doc_ids[i], float(sims[i])) for i in top]
        if window is not None:
            s, e = window
            return pairs[s:e]
        return pairs

    def score_pairs(self, query: str, doc_ids: list[str]) -> dict[str, float]:
        """Cosine of specific doc_ids against the query (for backfilling text_score)."""
        qv = self.encode_query(query)
        row = {d: i for i, d in enumerate(self.doc_ids)}
        out = {}
        for d in doc_ids:
            i = row.get(d)
            if i is not None:
                out[d] = float(np.asarray(self.doc_vectors[i], dtype=np.float32) @ qv)
        return out
