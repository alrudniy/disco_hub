"""
dh2.teachers -- the two cross-encoder teachers that produce the primary relevance
scores (MarginMSE / listwise targets). Both normalize to [0,1] so agreement logic and
grading thresholds are comparable across teachers.

  BgeTeacher   : BAAI/bge-reranker-v2-m3 via sentence_transformers.CrossEncoder.
                 bge-reranker-v2-m3 already outputs ~[0,1]; we sigmoid-guard stray logits.
  QwenTeacher  : Qwen3-Reranker-8B. This model scores relevance as the probability of a
                 "yes" token given an instruction+query+document prompt. We implement the
                 documented yes/no-logit -> probability readout, batched, on the H100.

Both expose:  .score(pairs: list[(query, doc_text)]) -> list[float] in [0,1]
Heavy deps imported lazily so this module imports fine on a CPU box.
"""
from __future__ import annotations

import math
from typing import Sequence

from dh2 import config2 as C


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BgeTeacher:
    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int | None = None, max_length: int | None = None):
        self.model_name = model_name or C.TEACHER_BGE
        self.device = device
        self.batch_size = batch_size or C.TEACHER.bge_batch_size
        self.max_length = max_length or C.TEACHER.rerank_max_length
        self._ce = None

    def _load(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.model_name, max_length=self.max_length,
                                    device=self.device)
        return self._ce

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        ce = self._load()
        raw = ce.predict(list(pairs), batch_size=self.batch_size,
                         show_progress_bar=False)
        out = []
        for s in raw:
            s = float(s)
            out.append(s if 0.0 <= s <= 1.0 else _sigmoid(s))
        return out


class QwenTeacher:
    """Qwen3-Reranker-8B relevance via yes-token probability (offline teacher, H100)."""

    _INSTRUCT = ("Given a pharmaceutical scouting query and a document, determine whether "
                 "the document is relevant and useful for the query.")

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int | None = None, max_length: int | None = None):
        self.model_name = model_name or C.TEACHER_QWEN
        self.device = device
        self.batch_size = batch_size or C.TEACHER.qwen_batch_size
        self.max_length = max_length or C.TEACHER.rerank_max_length
        self._tok = None
        self._model = None
        self._yes_id = None
        self._no_id = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16,
                device_map=self.device or "auto").eval()
            # token ids for yes/no (Qwen3-Reranker judges with these)
            self._yes_id = self._tok.convert_tokens_to_ids("yes")
            self._no_id = self._tok.convert_tokens_to_ids("no")
        return self._model

    def _format(self, query: str, doc: str) -> str:
        return (f"<Instruct>: {self._INSTRUCT}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc}\n"
                f"<Relevant>:")

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch
        model = self._load()
        tok = self._tok
        out: list[float] = []
        bs = self.batch_size
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            prompts = [self._format(q, d[: self.max_length * 6]) for q, d in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=self.max_length).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]   # last-token logits
            yes = logits[:, self._yes_id]
            no = logits[:, self._no_id]
            probs = torch.softmax(torch.stack([no, yes], dim=-1), dim=-1)[:, 1]
            out.extend([float(p) for p in probs.detach().cpu().tolist()])
        return out
