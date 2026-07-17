"""
dh2.teachers -- the two cross-encoder teachers that produce the primary relevance
scores (MarginMSE / listwise targets). Both normalize to [0,1] so agreement logic and
grading thresholds are comparable across teachers.

  BgeTeacher   : BAAI/bge-reranker-v2-m3 via sentence_transformers.CrossEncoder.
                 bge-reranker-v2-m3 already outputs ~[0,1]; we sigmoid-guard stray logits.
                 CAPPED AT 512 -- it is an XLM-R encoder and cannot go higher.
  QwenTeacher  : Qwen3-Reranker-8B, with the OFFICIAL chat framing, an 8,192-token
                 context, a pharmaceutical semantic instruction, and both the yes-token
                 probability and the raw yes-no logit margin.

Both expose:  .score(pairs: list[(query, doc_text)]) -> list[float] in [0,1]
QwenTeacher also exposes .score_detailed() -> [{"probability", "logit_margin"}].
Heavy deps imported lazily so this module imports fine on a CPU box.

A NOTE ON THE TWO TEACHERS (2026-07-15 measurements, n=55,605 adjudicated pairs):
BGE correlates +0.468 with query/doc word overlap but only +0.211 with the LLM's utility
grade; Qwen is +0.371 and +0.410. At FIXED relevance (grade 3), BGE's mean score swings
4.1x on wording alone (0.053 low-overlap vs 0.218 high) while Qwen's swings 1.15x (0.802
vs 0.921). BGE is a lexical matcher and 76% of this corpus's true matches are low-overlap,
so BGE is kept as an explicitly-labelled lexical channel, NOT as an equal semantic judge.
Qwen is the primary semantic reranker. Do not average their scores.
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


# The pharmaceutical semantic instruction (RASC S3). This is the load-bearing part of
# the reranker: the previous instruction ("determine whether the document is relevant and
# useful") gave the model no reason to treat B7-H1 and PD-L1 as the same target, and no
# reason to reject a document that shares vocabulary but not the requirement. Qwen already
# behaves this way when asked to (corr with LLM utility grade +0.410 vs BGE's +0.211); the
# instruction is what makes the behavior explicit and stable.
PHARMA_INSTRUCTION = (
    "Given a pharmaceutical or biotechnology scouting need, judge whether the\n"
    "document is practically useful for satisfying that need.\n"
    "\n"
    "Consider disease and disease stage, target or pathway, mechanism of action,\n"
    "modality, route of administration, patient population or biomarker,\n"
    "development stage, and licensing or commercial relevance.\n"
    "\n"
    "Treat scientifically justified legacy names, gene/protein aliases, drug\n"
    "development codes, class names, and patent/legal terminology as semantic\n"
    "equivalents. Do not reward shared words alone. Reject documents that match\n"
    "vocabulary but not the underlying scientific or commercial requirement."
)

# Official Qwen3-Reranker chat framing. Reproduced verbatim from the model card: the
# system turn, the assistant turn, and the empty <think> block are part of the format the
# checkpoint was trained on. The previous implementation invented its own plain-text
# prompt ("<Relevant>:") and sliced documents by character count before tokenizing, which
# put the model off-distribution and truncated the query/instruct along with the document.
QWEN_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. The answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


class QwenTeacher:
    """Qwen3-Reranker-8B relevance via yes/no logit readout (offline teacher + the RASC
    reranker).

    Returns probability in [0,1] from .score() so the existing teacher-merge/grading code
    keeps working unchanged. .score_detailed() additionally returns the raw logit margin,
    which is what the cascade RANKS on: softmax saturates near 1.0 (grade-3 mean 0.802 on
    low-overlap, 0.921 on high) and destroys the ordering information at the top of the
    list -- exactly where a top-10 shortlist is decided. The margin does not saturate.
    """

    _INSTRUCT = PHARMA_INSTRUCTION

    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int | None = None, max_length: int | None = None,
                 instruction: str | None = None, use_flash_attention: bool | None = None):
        self.model_name = model_name or C.TEACHER_QWEN
        self.device = device
        self.batch_size = batch_size or C.TEACHER.qwen_batch_size
        # 8,192 by default (not the BGE-bound 512). See config2.TeacherConfig.
        self.max_length = max_length or C.TEACHER.qwen_rerank_max_length
        self.instruction = instruction or self._INSTRUCT
        self.use_flash_attention = (C.TEACHER.qwen_use_flash_attention
                                    if use_flash_attention is None else use_flash_attention)
        self._tok = None
        self._model = None
        self._yes_id = None
        self._no_id = None
        self._prefix_ids: list[int] | None = None
        self._suffix_ids: list[int] | None = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tok = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            kwargs = {"torch_dtype": torch.bfloat16, "device_map": self.device or "auto"}
            if self.use_flash_attention:
                try:
                    import flash_attn  # noqa: F401
                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    # FA2 is an optimization, not a correctness requirement. A box without
                    # it must still produce identical scores, just slower -- so warn and
                    # fall back rather than crash a 5-hour unattended run.
                    print("[QwenTeacher] flash_attn not available; using default attention")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                               **kwargs).eval()
            # "yes"/"no" as the model tokenizes them in assistant position.
            self._yes_id = self._tok.convert_tokens_to_ids("yes")
            self._no_id = self._tok.convert_tokens_to_ids("no")
            if self._yes_id is None or self._no_id is None:
                raise RuntimeError("Qwen tokenizer has no yes/no ids -- wrong checkpoint?")
            self._prefix_ids = self._tok.encode(QWEN_PREFIX, add_special_tokens=False)
            self._suffix_ids = self._tok.encode(QWEN_SUFFIX, add_special_tokens=False)
        return self._model

    def _content(self, query: str, doc: str) -> str:
        return (f"<Instruct>: {self.instruction}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc}")

    def _encode_batch(self, chunk: Sequence[tuple[str, str]]):
        """Tokenize with truncation='longest_first' and EXPLICIT prefix/suffix.

        The prefix/suffix are appended after truncation, so the chat framing and the
        assistant turn can never be truncated away -- the old code truncated the whole
        formatted string, which on a long document silently removed the very tokens the
        yes/no readout depends on.
        """
        tok = self._tok
        budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        if budget <= 0:
            raise ValueError(f"max_length={self.max_length} too small for chat framing")
        texts = [self._content(q, d) for q, d in chunk]
        enc = tok(texts, truncation=True, max_length=budget, padding=False,
                  add_special_tokens=False)
        ids = [self._prefix_ids + x + self._suffix_ids for x in enc["input_ids"]]
        return tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")

    def score_detailed(self, pairs: Sequence[tuple[str, str]]) -> list[dict]:
        """Return [{"probability": float, "logit_margin": float}] per pair."""
        import torch
        model = self._load()
        out: list[dict] = []
        bs = self.batch_size
        for i in range(0, len(pairs), bs):
            chunk = list(pairs[i:i + bs])
            enc = self._encode_batch(chunk).to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]   # next-token logits
            yes = logits[:, self._yes_id].float()
            no = logits[:, self._no_id].float()
            margin = yes - no
            prob = torch.softmax(torch.stack([no, yes], dim=-1), dim=-1)[:, 1]
            for m, p in zip(margin.cpu().tolist(), prob.cpu().tolist()):
                out.append({"probability": float(p), "logit_margin": float(m)})
        return out

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Probability in [0,1] -- the teacher-merge interface, unchanged."""
        return [d["probability"] for d in self.score_detailed(pairs)]
