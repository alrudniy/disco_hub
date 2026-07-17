"""
Pluggable embedding backends.

Two implementations behind one interface. The interface is ASYMMETRIC because the
production encoder (Qwen3-Embedding) is: queries carry a task instruction,
documents do not.

    .encode_documents(texts)   passage side -- raw text, NO instruction
    .encode_queries(texts)     query side  -- "Instruct: {task}\nQuery: " prefix
    .encode(texts)             alias for encode_documents (back-compat)

Backends:

    MockEmbedder                deterministic, hash-seeded vectors. No GPU, no
                                downloads, identical across machines -- the tool
                                that validates plumbing and *proves* embedding
                                reproducibility in stage 09.
    SentenceTransformerEmbedder real Qwen3-Embedding-0.6B (or any ST model).
                                Lazy-imported so the package works without torch.

Why the asymmetry matters here: a research interest is phrased in clinical/
business language while a patent is phrased in legal/technical language. The
Qwen3 query instruction is the mechanism that bridges that register gap; encoding
queries WITHOUT it (the previous behavior) leaves measurable retrieval quality on
the table. See https://huggingface.co/Qwen/Qwen3-Embedding-0.6B for the exact
prompt contract this mirrors.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .config import EMBED_DIM, EMBED_MODEL, QUERY_INSTRUCTION


# --------------------------------------------------------------------------- #
# Query-instruction formatting (Qwen3-Embedding contract). Pure + unit-tested.
# --------------------------------------------------------------------------- #
def query_prompt_prefix(task: str) -> str:
    """
    The PREFIX that Sentence-Transformers prepends to each query string. ST does
    `prompt + text`, so the model ultimately sees "Instruct: {task}\nQuery: {q}".
    Documents are NOT given a prefix.
    """
    return f"Instruct: {task}\nQuery: "


def format_query(query: str, task: str) -> str:
    """The full instructed query string the model sees (prefix + query). Kept as a
    separate pure function so the exact contract is testable without the model."""
    return query_prompt_prefix(task) + query


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class MockEmbedder:
    """
    Deterministic embeddings: SHA-256(text) seeds a NumPy RNG that draws the
    vector, plus a hashed bag-of-tokens so texts sharing terminology land close.
    Identical text -> identical vector on any machine, every run.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(self.dim) * 0.05
        for tok in set(text.lower().split()):
            if len(tok) < 3:
                continue
            th = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
            base[th % self.dim] += 1.0
        return base

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))
        return _l2_normalize(mat)

    def encode_documents(self, texts: Sequence[str], batch_size=None) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str], batch_size=None) -> np.ndarray:
        # batch_size is accepted for interface parity but ignored: each mock vector
        # depends only on its own text, so the mock is batch-invariant BY
        # CONSTRUCTION. That is the honest mock answer to the batch-invariance test
        # (drift = 0); the real encoder below is where batch size can move vectors.
        # is a *learned-model* mechanism; a hash-of-tokens mock would treat the
        # instruction words ("instruct", "given", "retrieve", ...) as bag-of-words
        # noise and DILUTE the shared-term signal that makes mock retrieval behave
        # sanely. So the mock realizes the asymmetry structurally (07 calls
        # encode_queries, 04 calls encode_documents) but not numerically; the real
        # encoder below is where the instruction actually shapes the vector. This
        # keeps mock determinism and the stage-09 stability guarantees intact.
        return self.encode(texts)


class SentenceTransformerEmbedder:
    """Real encoder. Heavy deps imported lazily so import never fails in mock mode."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None,
                 batch_size: int = 64, query_instruction: str | None = None,
                 model_kwargs: dict | None = None,
                 tokenizer_kwargs: dict | None = None):
        from sentence_transformers import SentenceTransformer  # lazy

        # On Anvil GPUs the model card recommends flash-attention-2 + left padding
        # (the model uses last-token pooling). Pass those through, e.g.:
        #   model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"}
        #   tokenizer_kwargs={"padding_side": "left"}
        extra = {}
        if model_kwargs:
            extra["model_kwargs"] = model_kwargs
        if tokenizer_kwargs:
            extra["tokenizer_kwargs"] = tokenizer_kwargs
        self.model = SentenceTransformer(model_name, device=device, **extra)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = query_prompt_prefix(query_instruction or QUERY_INSTRUCTION)

    def _encode(self, texts: Sequence[str], prompt: str | None = None,
                batch_size: int | None = None) -> np.ndarray:
        kw = {"prompt": prompt} if prompt else {}
        vecs = self.model.encode(
            list(texts), batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine == dot product downstream
            show_progress_bar=False, **kw)
        return vecs.astype(np.float32)

    def encode_documents(self, texts: Sequence[str], batch_size=None) -> np.ndarray:
        # Passage side: NO instruction (Qwen3-Embedding contract).
        return self._encode(texts, batch_size=batch_size)

    def encode_queries(self, texts: Sequence[str], batch_size=None) -> np.ndarray:
        # Query side: prepend "Instruct: {task}\nQuery: ". ST concatenates the
        # prompt with each text, yielding the documented instructed-query format.
        return self._encode(texts, prompt=self.query_prefix, batch_size=batch_size)

    # Back-compat: bare .encode() means the document/passage side.
    def encode(self, texts: Sequence[str], batch_size=None) -> np.ndarray:
        return self.encode_documents(texts, batch_size=batch_size)


def get_embedder(mock: bool = True, **kwargs) -> Embedder:
    """Factory. mock=True -> MockEmbedder; mock=False -> real ST model. Real-mode
    kwargs (device, batch_size, query_instruction, model_kwargs, tokenizer_kwargs)
    pass through to SentenceTransformerEmbedder."""
    if mock:
        return MockEmbedder(dim=kwargs.get("dim", EMBED_DIM))
    kwargs.pop("dim", None)  # not a real-encoder argument
    return SentenceTransformerEmbedder(**kwargs)
